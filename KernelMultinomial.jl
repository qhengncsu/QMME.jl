function loglikelihood(Y::Matrix, P::Matrix)
	T = eltype(P)
	obj = zero(T)
	for i in axes(Y, 1)
		for j in axes(Y, 2)
			if Y[i,j] == one(T)
				obj += log(P[i,j])
			end
		end
	end
	obj
end

function solve_sylvester!(B::Matrix, RHS::Matrix, M::Real, ϵ::Real, e1::Vector,U::Matrix,ρ::Real,e2::Vector, 
                   V::Matrix, Ve2, buffer1::Matrix, buffer2::Matrix)
	(p, q)=size(B)
	mul!(buffer1,Transpose(U),RHS)
	mul!(buffer2,buffer1,Ve2)
	for j in 1:q
		for i in 1:p
			buffer2[i,j] = buffer2[i,j]/(e1[i]*M+e2[j]*ρ+ϵ)
		end
	end
	mul!(buffer1,U,buffer2)
	mul!(B,buffer1,Transpose(V))
end
    
function solve_sylvester!(X::Matrix, Xprime::Matrix, RHS::Matrix, ρ::Real, e2::Vector, Aprime::Matrix, 
                   U::Matrix, V::Matrix, buffer1::Matrix, buffer2::Matrix)
	(p, q)=size(X)
	mul!(buffer1,Transpose(U),RHS)
	mul!(buffer2,buffer1,V)
	for l in 1:q
		for k in 1:p
			rhs = buffer2[k,l]
			for j in 1:(k-1)
				rhs -= Aprime[k,j]*Xprime[j,l]
			end
			#for i in 1:(l-1)
			#	rhs -= Xprime[k,i]*Bprime[i,l]
			#end
			Xprime[k,l] = rhs/(Aprime[k,k]+ρ*e2[l])
		end
	end
	mul!(buffer1,U,Xprime)
	mul!(X,buffer1,Transpose(V))
end
    
function compute_probs_reduced!(P::Matrix{T}, ηs::Matrix{T}, row_sums::Vector{T}) where T <: AbstractFloat
	n, q = size(ηs)
	for i in 1:n, j in 1:q
		P[i,j] = exp(ηs[i,j])
	end
	fill!(row_sums, zero(T))
	@inbounds for i in 1:n, j in 1:q
		row_sums[i] += P[i,j]
	end
	for i in 1:n
		row_sums[i] += one(T)
		inv_sum = inv(row_sums[i])
		for j in 1:q
			P[i,j] *= inv_sum
		end
		P[i, q+1] = inv_sum
	end
end

function compute_probs!(P::Matrix, ηs::Matrix, row_sums::Vector)
	n, q = size(ηs)
	@inbounds for i in 1:n, j in 1:q
		P[i,j] = exp(ηs[i,j])
	end
	fill!(row_sums, zero(eltype(P)))
	@inbounds for i in 1:n, j in 1:q
		row_sums[i] += P[i,j]
	end  
	@inbounds for i in 1:n
		inv_sum = inv(row_sums[i])
		for j in 1:q
			P[i,j] *= inv_sum
		end
	end
end

function FastMultinomial_reduced(X::Matrix, Y::Matrix; tol::Real=1e-6, verbose::Bool=true, max_iters::Int=1000)
	(n, p), q = size(X), size(Y,2)
	ηs = zeros(n, q-1)
	r = zeros(n, q-1)
	P = similar(Y)
	row_sums_buffer = zeros(n)
	XTX, XTY = X'X + 1e-9*I , X'Y[:,1:(q-1)]
	L = cholesky!(Symmetric(XTX))
	B = zeros(p,q-1)
	direction, grad = similar(B), similar(B)
	mul!(ηs, X, B) # ηs = X * β
	obj, old_obj = 0.0, Inf
	(γ, δ) = (copy(B), copy(B))
	nesterov = 0 
	E = 2 .*(Diagonal(ones(q-1)) + ones(q-1)*Transpose(ones(q-1)))
	for iteration = 1:max_iters
		nesterov += 1
		@. B = γ + ((nesterov - 1)/(nesterov + 2)) * (γ - δ)
		@. δ = γ # Nesterov acceleration
		mul!(ηs, X, B) # ηs = X * β
		compute_probs_reduced!(P, ηs, row_sums_buffer)
		obj = -loglikelihood(Y, P)
		if verbose
			println(iteration, " ", obj, " ", nesterov)
		end
		if old_obj < obj 
			nesterov = 0 
		end
		@. r = Y[:,1:(q-1)] - P[:,1:(q-1)]
		mul!(grad, transpose(X), r)
		ldiv!(XTY, L, grad)
		mul!(direction, XTY, E)
		@. B = B + direction
		if abs(obj - old_obj) < tol * (abs(old_obj) + 1)
			return (B, obj, iteration)
		else
			@. γ = B
			old_obj = obj
		end
	end
	return (B, obj, max_iters)
end

function KernelMultinomial(Y::Matrix, K::Matrix, G::AbstractMatrix, λ::Real;
                    tol::Real=1e-4, verbose=true, ϵ=1e-4, adaptive::Bool=true, max_iters::Int=1000, Binit = nothing)
	n, q = size(K,1), size(Y, 2)
	ηs, r = zeros(n,q-1), zeros(n,q-1)
	E = 2 .*(Diagonal(ones(q-1)) + ones(q-1)*Transpose(ones(q-1)))
	(e2,V) = eigen(E)
	m = size(G,1)
    B = zeros(m,q-1)
    if Binit != nothing
        B .= Binit
    end
	direction, grad = similar(B),similar(B)
	buffer1, buffer2 = similar(B),similar(B)
	Bprime = similar(B)
	row_sums_buffer = zeros(n)
	KGT = K*Transpose(G)
	GKTKGT = Transpose(KGT)*KGT
	GKGT = G*KGT
	L = cholesky(GKGT+ϵ*I)
	LeftM = L \ GKTKGT
	SchurAt = schur(transpose(LeftM))
	Aprime = Matrix(transpose(SchurAt.T))
	U = SchurAt.Z
	RHS = similar(grad)
	P = zeros(n,q)
	nesterov = 0
	(obj, old_obj) = (0.0, Inf)
	(γ, δ) = (copy(B), copy(B))
	buffer3 = zeros(q-1,q-1)
	objs = zeros(max_iters)
	grad_norms = zeros(max_iters)
	times = zeros(max_iters)
	start_time = time()
	for iter = 1:max_iters
		nesterov += 1
		if adaptive
			@. B = γ + (nesterov /(nesterov + 2)) * (γ - δ)
		else
			@. B = γ + 0.333 * (γ - δ)
		end
		@. δ = γ # Nesterov acceleration
		mul!(ηs, KGT, B) # ηs = X * B
		compute_probs_reduced!(P,ηs,row_sums_buffer)
		obj = -loglikelihood(Y, P)
		mul!(buffer1, GKGT, B)
		mul!(buffer3, Transpose(B), buffer1)
		obj += 0.5*λ*tr(buffer3)
		@. r = Y[:,1:(q-1)] - P[:,1:(q-1)]
		mul!(grad, Transpose(KGT), r)
		@. grad *= -1
		@. grad += λ*buffer1
		norm_grad = norm(grad,2)
		if verbose
			println(iter," ",norm_grad, " ", obj," ",nesterov)
		end
		objs[iter] = obj
		grad_norms[iter] = norm_grad
		times[iter] = time()-start_time
		ldiv!(RHS,L,grad)
		mul!(grad,RHS,E)
		#solve_sylvester!(direction,RHS,1.0,ϵ,e1,U,λ,e2,V,Ve2,buffer1,buffer2)
		#direction .= sylvester(LeftM, λ*E,grad)
		solve_sylvester!(direction, Bprime, grad, λ, e2, Aprime, U, V, buffer1, buffer2)
		@. B = B - direction
		if old_obj < obj 
			nesterov = 0
		end
		if norm_grad<tol
			return (B,objs[1:iter], grad_norms[1:iter], times[1:iter])
		else
			@. γ = B
			old_obj = obj
		end
	end
	return (B,objs, grad_norms, times)
end

function KernelMultinomial_full(Y::Matrix, K::Matrix, G::AbstractMatrix, λ::Real;
                      tol::Real=1e-4, verbose=true, ϵ=1e-4, adaptive::Bool=true, max_iters::Int=1000, Binit=nothing)
	n, q = size(K,1), size(Y, 2)
	ηs, r = zeros(n,q), zeros(n,q)
	E = 2 .*(Diagonal(ones(q)) + ones(q)*Transpose(ones(q)))
	(e2,V) = eigen(E)
	m = size(G,1)
	B = zeros(m,q)
    if Binit != nothing
        B .= Binit
    end
	direction, grad = similar(B),similar(B)
	buffer1, buffer2 = similar(B),similar(B)
	Bprime = similar(B)
	row_sums_buffer = zeros(n)
	KGT = K*Transpose(G)
	GKTKGT = Transpose(KGT)*KGT
	GKGT = G*KGT
	L = cholesky(GKGT+ϵ*I)
	LeftM = L \ GKTKGT
	SchurAt = schur(transpose(LeftM))
	Aprime = Matrix(transpose(SchurAt.T))
	U = SchurAt.Z
	RHS = similar(grad)
	P = zeros(n,q)
	nesterov = 0
	(obj, old_obj) = (0.0, Inf)
	(γ, δ) = (copy(B), copy(B))
	buffer3 = zeros(q,q)
	objs = zeros(max_iters)
	grad_norms = zeros(max_iters)
	times = zeros(max_iters)
	start_time = time()
	for iter = 1:max_iters
		nesterov += 1
		if adaptive
			@. B = γ + (nesterov /(nesterov + 2)) * (γ - δ)
		else
			@. B = γ + 0.333 * (γ - δ)
		end
		@. δ = γ # Nesterov acceleration
		mul!(ηs, KGT, B) # ηs = X * B
		compute_probs!(P,ηs,row_sums_buffer)
		obj = -loglikelihood(Y, P)
		mul!(buffer1, GKGT, B)
		mul!(buffer3, Transpose(B), buffer1)
		obj += 0.5*λ*tr(buffer3)
		@. r = Y - P
		mul!(grad, Transpose(KGT), r)
		@. grad *= -1
		@. grad += λ*buffer1
		norm_grad = norm(grad,2)
		if verbose
			println(iter," ",norm_grad, " ", obj," ",nesterov)
		end
		objs[iter] = obj
		grad_norms[iter] = norm_grad
		times[iter] = time()-start_time
		ldiv!(RHS,L,grad)
		mul!(grad,RHS,E)
		#solve_sylvester!(direction,RHS,1.0,ϵ,e1,U,λ,e2,V,Ve2,buffer1,buffer2)
		#direction .= sylvester(LeftM, λ*E,grad)
		solve_sylvester!(direction, Bprime, grad, λ, e2, Aprime, U, V, buffer1, buffer2)
		@. B = B - direction
		if old_obj < obj 
			nesterov = 0
		end
		if norm_grad<tol
			return (B,objs[1:iter], grad_norms[1:iter], times[1:iter])
		else
			@. γ = B
			old_obj = obj
		end
	end
	return (B,objs, grad_norms, times)
end

function KernelMultinomial(Y::Matrix, K::Matrix, G::Matrix, λs::Vector;
                    tol::Real=1e-4, verbose=true, ϵ=1e-4, adaptive::Bool=true, max_iters::Int=1000)
	start_time = time()
	n, q = size(K,1), size(Y, 2)
	ηs, r = zeros(n,q-1), zeros(n,q-1)
	E = 2 .*(Diagonal(ones(q-1)) + ones(q-1)*Transpose(ones(q-1)))
	(e2,V) = eigen(E)
	m = size(G,1)
	B = zeros(m,q-1)
	direction, grad = similar(B),similar(B)
	buffer1, buffer2 = similar(B),similar(B)
	Bprime = similar(B)
	row_sums_buffer = zeros(n)
	KGT = K*Transpose(G)
	GKTKGT = Transpose(KGT)*KGT
	GKGT = G*KGT
	L = cholesky(GKGT+ϵ*I)
	LeftM = L \ GKTKGT
	SchurAt = schur(transpose(LeftM))
	Aprime = Matrix(transpose(SchurAt.T))
	U = SchurAt.Z
	RHS = similar(grad)
	P = zeros(n,q)
	nesterov = 0
	buffer3 = zeros(q-1,q-1)
	Bs = zeros(length(λs),m,q-1)
	for j in 1:length(λs)
		(obj, old_obj) = (0.0, Inf)
		(γ, δ) = (copy(B), copy(B))
		total_iter = 0
		λ = λs[j]
		nesterov = 0
		for iter = 1:max_iters
			nesterov += 1
			if adaptive
				@. B = γ + (nesterov /(nesterov + 2)) * (γ - δ)
			else
				@. B = γ + 0.333 * (γ - δ)
			end
			@. δ = γ # Nesterov acceleration
			mul!(ηs, KGT, B) # ηs = X * B
			compute_probs_reduced!(P,ηs,row_sums_buffer)
			obj = -loglikelihood(Y, P)
			mul!(buffer1, GKGT, B)
			mul!(buffer3, Transpose(B), buffer1)
			obj += 0.5*λ*tr(buffer3)
			@. r = Y[:,1:(q-1)] - P[:,1:(q-1)]
			mul!(grad, Transpose(KGT), r)
			@. grad *= -1
			@. grad += λ*buffer1
			norm_grad = norm(grad,2)
			if verbose
				println(iter," ",norm_grad, " ", obj," ",nesterov)
			end
			ldiv!(RHS,L,grad)
			mul!(grad,RHS,E)
			#solve_sylvester!(direction,RHS,1.0,ϵ,e1,U,λ,e2,V,Ve2,buffer1,buffer2)
			#direction .= sylvester(LeftM, λ*E,grad)
			solve_sylvester!(direction, Bprime, grad, λ, e2, Aprime, U, V, buffer1, buffer2)
			@. B = B - direction
			if old_obj < obj 
				nesterov = 0
			end
			if norm_grad<tol
				total_iter = iter
				break
			else
				@. γ = B
				old_obj = obj
			end
			total_iter = max_iters
		end
		Bs[j,:,:] .= B
		println("λ=",λ, " sovled in ", total_iter ," iterations.")
	end
	return Bs, time()-start_time
end
    
function KernelMultinomial_full(Y::Matrix, K::Matrix, G::Matrix, λs::Vector;
                     tol::Real=1e-4, verbose::Bool=true, ϵ::Real=1e-4, 
                     adaptive::Bool=true, max_iters::Int=1000)
	start_time = time()
	n, q = size(K,1), size(Y, 2)
	ηs, r = zeros(n,q), zeros(n,q)
	E = 2 .*(Diagonal(ones(q)) + ones(q)*Transpose(ones(q)))
	(e2,V) = eigen(E)
	m = size(G,1)
	B = zeros(m,q)
	direction, grad = similar(B),similar(B)
	buffer1, buffer2 = similar(B),similar(B)
	Bprime = similar(B)
	row_sums_buffer = zeros(n)
	KGT = K*Transpose(G)
	GKTKGT = Transpose(KGT)*KGT
	GKGT = G*KGT
	L = cholesky(GKGT+ϵ*I)
	LeftM = L \ GKTKGT
	SchurAt = schur(transpose(LeftM))
	Aprime = Matrix(transpose(SchurAt.T))
	U = SchurAt.Z
	RHS = similar(grad)
	P = zeros(n,q)
	nesterov = 0
	(obj, old_obj) = (0.0, Inf)
	(γ, δ) = (copy(B), copy(B))
	buffer3 = zeros(q,q)
	Bs = zeros(length(λs),m,q)
	for j in 1:length(λs)
		(obj, old_obj) = (0.0, Inf)
		(γ, δ) = (copy(B), copy(B))
		total_iter = 0
		λ = λs[j]
		nesterov = 0
		for iter = 1:max_iters
			nesterov += 1
			if adaptive
				@. B = γ + (nesterov /(nesterov + 2)) * (γ - δ)
			else
				@. B = γ + 0.333 * (γ - δ)
			end
			@. δ = γ # Nesterov acceleration
			mul!(ηs, KGT, B) # ηs = X * B
			compute_probs!(P,ηs,row_sums_buffer)
			obj = -loglikelihood(Y, P)
			mul!(buffer1, GKGT, B)
			mul!(buffer3, Transpose(B), buffer1)
			obj += 0.5*λ*tr(buffer3)
			@. r = Y - P
			mul!(grad, Transpose(KGT), r)
			@. grad *= -1
			@. grad += λ*buffer1
			norm_grad = norm(grad,2)
			if verbose
				println(iter," ",norm_grad, " ", obj," ",nesterov)
			end
			ldiv!(RHS,L,grad)
			mul!(grad,RHS,E)
			#solve_sylvester!(direction,RHS,1.0,ϵ,e1,U,λ,e2,V,Ve2,buffer1,buffer2)
			#direction .= sylvester(LeftM, λ*E,grad)
			solve_sylvester!(direction, Bprime, grad, λ, e2, Aprime, U, V, buffer1, buffer2)
			@. B = B - direction
			if old_obj < obj 
				nesterov = 0
			end
			if norm_grad<tol
				total_iter = iter
				break
			else
				@. γ = B
				old_obj = obj
			end
			total_iter = max_iters
		end
		Bs[j,:,:] .= B
		println("λ=",λ, " sovled in ", total_iter ," iterations.")
	end
	return Bs, time()-start_time
end


function KernelMultinomial_newton(Y::Matrix, K::Matrix, G::Matrix, λ::Real;
                         tol::Real=1e-4, verbose::Bool=true, ϵ::Real=1e-4, max_iters::Int=100)
	n, q = size(K,1), size(Y, 2)
	ηs, r = zeros(n,q-1), zeros(n,q-1)
	m = size(G,1)
	B = zeros(m,q-1)
	direction, grad = similar(B), similar(B)
	direction_vec = zeros(m*(q-1))
	buffer1, buffer2 = similar(B), similar(B)
	row_sums_buffer = zeros(n)
	P = zeros(n,q)
	(obj, old_obj) = (0.0, Inf)
	buffer3 = zeros(q-1,q-1)
	objs = zeros(max_iters)
	grad_norms = zeros(max_iters)
	times = zeros(max_iters)
	H = zeros(m*(q-1), m*(q-1))
	KGT = K*Transpose(G)
	GKGT = G*KGT
	GKGTeps = GKGT + ϵ*I
	H1 = similar(H)
	H2 = similar(H)
	block = zeros(m,m)
	scaled_KGT = copy(KGT)
	start_time = time()
	for iter = 1:max_iters
		times[iter] = time() - start_time
		mul!(ηs, KGT, B) # ηs = X * B
		compute_probs_reduced!(P, ηs, row_sums_buffer)
		old_obj = -loglikelihood(Y, P)
		mul!(buffer1, GKGT, B)
		mul!(buffer3, Transpose(B), buffer1)
		old_obj += 0.5 * λ * tr(buffer3)
		@. r = Y[:, 1:(q-1)] - P[:, 1:(q-1)]
		mul!(grad, Transpose(KGT), r)
		@. grad *= -1
		@. grad += λ * buffer1
		norm_grad = norm(grad, 2)
		fill!(H1, 0.0)
		fill!(H2, 0.0)
		for k in 1:(q-1)
			scaled_KGT .= P[:, k] .* KGT
			mul!(block, Transpose(KGT), scaled_KGT)
			rows = ((k-1)*m + 1):k*m
			H1[rows, rows] .= block
		end
		for l in 1:(q-1)
			for k in 1:(q-1)
				p_kp_l = P[:, k] .* P[:, l]
				scaled_KGT .= p_kp_l .* KGT
				mul!(block, Transpose(KGT), scaled_KGT)
				rows = ((k-1)*m + 1):k*m
				cols = ((l-1)*m + 1):l*m
				H2[rows, cols] .= block
			end
		end
		H .= H1 - H2
		for j in 1:(q-1)
			H[((j-1)*m+1):(j*m), ((j-1)*m+1):(j*m)] .+= λ * GKGTeps
		end
		direction_vec .= H \ vec(grad)
		direction .= reshape(direction_vec, m, q-1)
		t = 1
		stephalving_count = 0
		while stephalving_count <= 20
			@. B -= t * direction
			mul!(ηs, KGT, B) # ηs = X * B
			compute_probs_reduced!(P, ηs, row_sums_buffer)
			obj = -loglikelihood(Y, P)
			mul!(buffer1, GKGT, B)
			mul!(buffer3, Transpose(B), buffer1)
			obj += 0.5 * λ * tr(buffer3)
			if obj < old_obj
				break
			else
				stephalving_count += 1
				@. B += t * direction
				t *= 0.5
			end
		end
		if stephalving_count == 21
			@. B -= direction
		end
		objs[iter] = old_obj
		grad_norms[iter] = norm_grad
		if verbose
			println(iter, " ", norm_grad, " ", old_obj, " ", stephalving_count)
		end
		if norm_grad < tol
			return (B, objs[1:iter], grad_norms[1:iter], times[1:iter])
		end
	end
	return (B, objs, grad_norms, times)
end

function KernelMultinomial_newton(Y::Matrix, K::Matrix, G::Matrix, λs::Vector;
                         tol::Real=1e-4, verbose::Bool=true, ϵ::Real=1e-4, max_iters::Int=100)
	start_time = time()
	n, q = size(K,1), size(Y, 2)
	ηs, r = zeros(n,q-1), zeros(n,q-1)
	m = size(G,1)
	B = zeros(m,q-1)
	direction, grad = similar(B), similar(B)
	direction_vec = zeros(m*(q-1))
	buffer1, buffer2 = similar(B), similar(B)
	row_sums_buffer = zeros(n)
	P = zeros(n,q)
	(obj, old_obj) = (0.0, Inf)
	buffer3 = zeros(q-1,q-1)
	objs = zeros(max_iters)
	grad_norms = zeros(max_iters)
	times = zeros(max_iters)
	H = zeros(m*(q-1), m*(q-1))
	KGT = K*Transpose(G)
	GKGT = G*KGT
	GKGTeps = GKGT + ϵ*I
	Bs = zeros(length(λs), m, q-1)
	H1 = similar(H)
	H2 = similar(H)
	block = zeros(m,m)
	scaled_KGT = copy(KGT)
	for j in 1:length(λs)
		(obj, old_obj) = (0.0, Inf)
		total_iter = 0
		λ = λs[j]
		for iter = 1:max_iters
			mul!(ηs, KGT, B) # ηs = X * B
			compute_probs_reduced!(P, ηs, row_sums_buffer)
			old_obj = -loglikelihood(Y, P)
			mul!(buffer1, GKGT, B)
			mul!(buffer3, Transpose(B), buffer1)
			old_obj += 0.5 * λ * tr(buffer3)
			@. r = Y[:, 1:(q-1)] - P[:, 1:(q-1)]
			mul!(grad, Transpose(KGT), r)
			@. grad *= -1
			@. grad += λ * buffer1
			norm_grad = norm(grad, 2)
			fill!(H1, 0.0)
			fill!(H2, 0.0)
			for k in 1:(q-1)
				scaled_KGT .= P[:, k] .* KGT
				mul!(block, Transpose(KGT), scaled_KGT)
				rows = ((k-1)*m + 1):k*m
				H1[rows, rows] .= block
			end
			for l in 1:(q-1)
				for k in 1:(q-1)
					p_kp_l = P[:, k] .* P[:, l]
					scaled_KGT .= p_kp_l .* KGT
					mul!(block, Transpose(KGT), scaled_KGT)
					rows = ((k-1)*m + 1):k*m
					cols = ((l-1)*m + 1):l*m
					H2[rows, cols] .= block
				end
			end
			H .= H1 - H2
			for j in 1:(q-1)
				H[((j-1)*m+1):(j*m), ((j-1)*m+1):(j*m)] .+= λ * GKGTeps
			end
			direction_vec .= H \ vec(grad)
			direction .= reshape(direction_vec, m, q-1)
			t = 1
			stephalving_count = 0
			while stephalving_count <= 20
				@. B -= t * direction
				mul!(ηs, KGT, B) # ηs = X * B
				compute_probs_reduced!(P, ηs, row_sums_buffer)
				obj = -loglikelihood(Y, P)
				mul!(buffer1, GKGT, B)
				mul!(buffer3, Transpose(B), buffer1)
				obj += 0.5 * λ * tr(buffer3)
				if obj < old_obj
					break
				else
					stephalving_count += 1
					@. B += t * direction
					t *= 0.5
				end
			end
			if stephalving_count == 21
				@. B -= direction
			end
			if verbose
				println(iter, " ", norm_grad, " ", old_obj, " ", stephalving_count)
			end
			if norm_grad < tol
				total_iter = iter
				break
			end
			total_iter = max_iters
		end
		Bs[j, :, :] .= B
		println("λ=", λ, " sovled in ", total_iter, " iterations.")
	end
	return Bs, time() - start_time
end


function KernelMultinomial_fista(Y::Matrix, K::Matrix, G::Matrix, λ::Real;
                        tol::Real=1e-4, verbose::Bool=true, max_iters::Int=3000)
	n, q = size(K,1), size(Y, 2)
	ηs, r = zeros(n,q-1), zeros(n,q-1)
	m = size(G,1)
	B = zeros(m,q-1)
	direction, grad = similar(B), similar(B)
	buffer1, buffer2 = similar(B), similar(B)
	row_sums_buffer = zeros(n)
	P = zeros(n,q)
	(obj, old_obj) = (0.0, Inf)
	(γ, δ) = (copy(B), copy(B))
	buffer3 = zeros(q-1,q-1)
	objs = zeros(max_iters)
	grad_norms = zeros(max_iters)
	times = zeros(max_iters)
	KGT = K*Transpose(G)
	GKGT = G*KGT
	opnormKGT = opnorm(KGT)
	opnormGKGT = opnorm(GKGT)
	step_size = 1 / (opnormKGT^2 + λ*opnormGKGT)
	start_time = time()
	t = 1
	for iter = 1:max_iters
		times[iter] = time() - start_time
		t_next = (1 + sqrt(1 + 4*t^2)) / 2
		B = γ + ((t - 1) / t_next) * (γ - δ)
		@. δ = γ # Nesterov acceleration
		t = t_next
		mul!(ηs, KGT, B) # ηs = X * B
		compute_probs_reduced!(P, ηs, row_sums_buffer)
		obj = -loglikelihood(Y, P)
		mul!(buffer1, GKGT, B)
		mul!(buffer3, Transpose(B), buffer1)
		obj += 0.5 * λ * tr(buffer3)
		@. r = Y[:,1:(q-1)] - P[:,1:(q-1)]
		mul!(grad, Transpose(KGT), r)
		@. grad *= -1
		@. grad += λ * buffer1
		norm_grad = norm(grad, 2)
		if verbose
			println(iter, " ", norm_grad, " ", obj)
		end
		objs[iter] = obj
		grad_norms[iter] = norm_grad
		@. B = B - step_size * grad
		if norm_grad < tol
			return (B, objs[1:iter], grad_norms[1:iter], times[1:iter])
		else
			@. γ = B
			old_obj = obj
		end
	end
	return (B, objs, grad_norms, times)
end 

function KernelMultinomial_adagd(Y::Matrix, K::Matrix, G::Matrix, λ::Real;
                        tol::Real=1e-4, verbose::Bool=true, max_iters::Int=3000)
	n, q = size(K,1), size(Y, 2)
	ηs, r = zeros(n,q-1), zeros(n,q-1)
	m = size(G,1)
	B = zeros(m,q-1)
	KGT = K*Transpose(G)
	GKGT = G*KGT
	direction, grad = similar(B), similar(B)
	buffer1, buffer2 = similar(B), similar(B)
	row_sums_buffer = zeros(n)
	P = zeros(n,q)
	(obj, old_obj) = (0.0, Inf)
	(B_prev, grad_prev) = copy(B), copy(B)
	buffer3 = zeros(q-1,q-1)
	objs = zeros(max_iters)
	grad_norms = zeros(max_iters)
	times = zeros(max_iters)
	start_time = time()
	θ = Inf
	t = t_prev = 1e-7
	for iter = 1:max_iters
		mul!(ηs, KGT, B) # ηs = X * B
		compute_probs_reduced!(P, ηs, row_sums_buffer)
		obj = -loglikelihood(Y, P)
		mul!(buffer1, GKGT, B)
		mul!(buffer3, Transpose(B), buffer1)
		obj += 0.5 * λ * tr(buffer3)
		@. r = Y[:,1:(q-1)] - P[:,1:(q-1)]
		mul!(grad, Transpose(KGT), r)
		@. grad *= -1
		@. grad += λ * buffer1
		norm_grad = norm(grad, 2)
		if verbose
			println(iter, " ", norm_grad, " ", obj)
		end
		objs[iter] = obj
		grad_norms[iter] = norm_grad
		times[iter] = time() - start_time
		if norm_grad < tol
			return (B, objs[1:iter], grad_norms[1:iter], times)
		end
		if iter == 1
			@. B -= t * grad
			@. grad_prev = grad
		else
			t = min(sqrt(1+θ)*t_prev, norm(B - B_prev) / (2 * norm(grad - grad_prev)))
			@. B_prev = B
			@. grad_prev = grad
			@. B -= t * grad
			θ = t / t_prev
			t_prev = t
		end
	end
	return (B, objs, grad_norms, times)
end