function FastLogistic(X::Matrix, y::Vector; tol::Real=1e-6, verbose::Bool=true, max_iters::Int=1000)
	(n, p) = size(X)
	ηs, r = similar(y), similar(y)
	XTX, XTy = X'X, X'y
	L = cholesky!(Symmetric(XTX))
	β = zeros(p)
	direction, grad = similar(β), similar(β) # work vectors
	mul!(ηs, X, β) # ηs = X * β
	obj, old_obj = -loglikelihood(y, ηs), Inf
	(γ, δ) = (copy(β), copy(β))
	nesterov = 0 
	for iteration = 1:max_iters
		nesterov  += 1
		@. β = γ + ((nesterov  - 1)/(nesterov + 2)) * (γ - δ)
		@. δ = γ # Nesterov acceleration
		mul!(ηs, X, β) # ηs = X * β
		obj = -loglikelihood(y, ηs)
		if verbose
			println(iteration," ",obj," ",nesterov)
		end
		if old_obj < obj 
			nesterov = 0 
		end
		for i = 1:n
			r[i] = y[i] - 1/(1+exp(-ηs[i]))
		end
		mul!(grad, transpose(X), r)
		ldiv!(direction, L, grad)
		@. β = β + 4* direction
		if abs(old_obj - obj) < tol*(abs(old_obj)+1)
			return (β, obj, iteration)
		else
			@. γ = β
			old_obj = obj
		end 
	end
	return (β, obj, max_iters)
end


function loglikelihood(ys::Vector{T}, ηs::Vector{T}) where T <: AbstractFloat
	obj = zero(eltype(ηs))
	for i in eachindex(ys)
		obj += ys[i]*ηs[i] - log1p(exp(ηs[i]))
	end
	obj
end

function kernel_logistic(y::Vector, K::Matrix, G::AbstractMatrix, λ::Real;
	ϵ::Real=1e-9, verbose::Bool=true, tol::Real = 1e-4, adaptive::Bool=true, max_iters::Int=1000)
	n = length(y)
	m = size(G,1)
	(r, s) = zeros(n), zeros(n) # work vectors
	β = zeros(m)
	KGT = K*Transpose(G)
	GKGT = G*KGT
	direction = similar(β)
	grad = similar(β)
	old_obj = Inf
	(γ, δ) = (copy(β), copy(β))
	grad_res = zeros(m)
	nesterov = 0
	ηs = zeros(n)
	objs = zeros(max_iters)
	grad_norms = zeros(max_iters)
	times = zeros(max_iters)
	start_time = time()
	H = 0.25*Transpose(KGT)*KGT + λ*GKGT +ϵ*I
	L = cholesky!(H)
	for iter = 1:max_iters
		nesterov = nesterov + 1
		if adaptive
			@. β = γ + (nesterov /(nesterov + 2)) * (γ - δ)
		else
			@. β = γ + 0.333 * (γ - δ)
		end
		@. δ = γ # Nesterov acceleration
		mul!(ηs, KGT, β) # ηs = X * β
		for i = 1:n
			r[i] = y[i] - 1/(1+exp(-ηs[i]))
		end
		mul!(grad_res, Transpose(KGT), r) 
		@. grad_res *= -1
		obj = -loglikelihood(y,ηs)
		mul!(grad,GKGT,β)
		obj += 0.5*λ*dot(grad,β)
		@. grad *= λ
		@. grad += grad_res
		norm_grad = norm(grad, 2)
		if verbose
			println(iter,"  ",norm_grad,"  ", " ",obj, " ",nesterov)
		end
		objs[iter] = obj
		grad_norms[iter] = norm_grad
		times[iter] = time()-start_time
		if norm_grad < tol && iter>1
			return (β,objs[1:iter], grad_norms[1:iter], times[1:iter])
		end
		ldiv!(direction, L, grad)
		@. β -= direction
		if old_obj < obj
			nesterov = 0 
		end # restart Nesterov momentum
		@. γ = β
		old_obj = obj
	end
	return (β,objs, grad_norms, times)
end

function kernel_logistic(y::Vector, K::Matrix, G::AbstractMatrix, λs::Vector;
                  ϵ::Real=1e-9, verbose::Bool=true, 
                  tol::Real = 1e-4, adaptive::Bool=true, max_iters::Int=1000)
	n = length(y)
	m = size(G,1)
	(r, s) = zeros(n), zeros(n) # work vectors
	β = zeros(m)
	KGT = K*Transpose(G)
	GKGT = G*KGT
	direction = similar(β)
	grad = similar(β)
	grad_res = zeros(m)
	nesterov = 0
	ηs = zeros(n)
	baseH = 0.25*Transpose(KGT)*KGT +ϵ*I
	H = copy(baseH)
	βs = zeros(length(λs),m)
	start_time = time()
	for j in 1:length(λs)
		(γ, δ) = (copy(β), copy(β))
		total_iter = 0
		λ = λs[j]
		nesterov = 0
		old_obj = Inf
		@. H = baseH + λ*GKGT
		total_iter = 0
		L = cholesky!(H)
		for iter = 1:max_iters
			nesterov = nesterov + 1
			if adaptive
				@. β = γ + (nesterov /(nesterov + 2)) * (γ - δ)
			else
				@. β = γ + 0.333 * (γ - δ)
			end
			@. δ = γ # Nesterov acceleration
			mul!(ηs, KGT, β) # ηs = X * β
			for i = 1:n
				r[i] = y[i] - 1/(1+exp(-ηs[i]))
			end
			mul!(grad_res, Transpose(KGT), r) 
			@. grad_res *= -1
			obj = -loglikelihood(y,ηs)
			mul!(grad,GKGT,β)
			obj += 0.5*λ*dot(grad,β)
			@. grad *= λ
			@. grad += grad_res
			norm_grad = norm(grad, 2)
			if verbose
				println(iter,"  ",norm_grad,"  ", " ",obj, " ",nesterov)
			end
			if norm_grad < tol && iter>1
				total_iter = iter
				break
			end
			ldiv!(direction, L, grad)
			@. β -= direction
			if old_obj < obj
				nesterov = 0 
			end # restart Nesterov momentum
			@. γ = β
			old_obj = obj
			total_iter = max_iters
		end
		βs[j,:] .= β
		println("λ=",λ, " sovled in ", total_iter ," iterations.")
	end
	return βs, time()-start_time
end

function kernel_logistic_newton(y::Vector, K::Matrix, G::AbstractMatrix, λ::Real;
                        verbose::Bool=true, tol::Real = 1e-4, ϵ::Real = 1e-9, max_iters::Int=100)
	n = length(y)
	m = size(G,1)
	(r, s) = zeros(n), zeros(n) # work vectors
	β = zeros(m)
	KGT = K*Transpose(G)
	GKGT = G*KGT
	direction = similar(β)
	grad_res = similar(β)
	old_obj = Inf
	grad = zeros(m)
	objs = zeros(max_iters)
	grad_norms = zeros(max_iters)
	times = zeros(max_iters)
	H1 = zeros(n,m)
	H = zeros(m,m)
	ws = zeros(n)
	diagM = Diagonal(ones(m))
	ηs = zeros(n)
	start_time = time()
	stephalving_count = 0
	for iter = 1:max_iters
		mul!(ηs, KGT, β) # ηs = X * β
		for i = 1:n
			ws[i] = 1/(1+exp(-ηs[i]))
			r[i] = y[i] - ws[i]
		end
		mul!(grad_res, Transpose(KGT), r) 
		@. grad_res *= -1
		old_obj = -loglikelihood(y,ηs)
		mul!(grad,GKGT,β)
		old_obj += 0.5*λ*dot(grad,β)
		@. grad *= λ
		@. grad += grad_res
		norm_grad = norm(grad, 2)
		objs[iter] = old_obj
		grad_norms[iter] = norm_grad
		times[iter] = time()-start_time
		if verbose
			println(iter,"  ",norm_grad,"  ", " ",old_obj, " ",stephalving_count)
		end
		if norm_grad < tol && iter>1
			return (β,objs[1:iter], grad_norms[1:iter], times[1:iter])
		end
		mul!(H1,Diagonal(ws .* (1 .-ws)),KGT)
		mul!(H, Transpose(KGT),H1)
		@. H += λ*GKGT + ϵ*diagM
		direction .= H\grad
		t = 1
		stephalving_count = 0
		while stephalving_count<= 20
			@. β -= t*direction
			mul!(ηs, KGT, β) # ηs = X * β
			obj = -loglikelihood(y,ηs)
			mul!(grad, GKGT ,β)
			obj += 0.5*λ*dot(grad,β)
			if obj < old_obj
				break
			else
				stephalving_count += 1
				@. β += t*direction
				t = t*0.5
			end
		end
		if stephalving_count==21
			@. β -= direction
		end
	end
	return (β,objs, grad_norms, times)
end

function kernel_logistic_newton(y::Vector, K::Matrix, G::AbstractMatrix, λs::Vector;
                        verbose::Bool=true, tol::Real = 1e-4, 
                        ϵ::Real = 1e-9, max_iters::Int=100)
	start_time = time()
	n = length(y)
	m = size(G,1)
	(r, s) = zeros(n), zeros(n) # work vectors
	β = zeros(m)
	KGT = K*Transpose(G)
	GKGT = G*KGT
	direction = similar(β)
	grad_res = similar(β)
	old_obj = Inf
	grad = zeros(m)
	H1 = zeros(n,m)
	H = zeros(m,m)
	ws = zeros(n)
	diagM = Diagonal(ones(m))
	ηs = zeros(n)
	start_time = time()
	stephalving_count = 0
	βs = zeros(length(λs),m)
	for j in 1:length(λs)
		total_iter = 0
		λ = λs[j]
		nesterov = 0
		old_obj = Inf
		for iter = 1:max_iters
			mul!(ηs, KGT, β) # ηs = X * β
			for i = 1:n
				ws[i] = 1/(1+exp(-ηs[i]))
				r[i] = y[i] - ws[i]
			end
			mul!(grad_res, Transpose(KGT), r) 
			@. grad_res *= -1
			old_obj = -loglikelihood(y,ηs)
			mul!(grad,GKGT,β)
			old_obj += 0.5*λ*dot(grad,β)
			@. grad *= λ
			@. grad += grad_res
			norm_grad = norm(grad, 2)
			if verbose
				println(iter,"  ",norm_grad,"  ", " ",old_obj, " ",stephalving_count)
			end
			if norm_grad < tol && iter>1
				total_iter = iter
				break
			end
			mul!(H1,Diagonal(ws .* (1 .-ws)),KGT)
			mul!(H, Transpose(KGT),H1)
			@. H += λ*GKGT + ϵ*diagM
			direction .= H\grad
			t = 1
			stephalving_count = 0
			while stephalving_count<= 20
				@. β -= t*direction
				mul!(ηs, KGT, β) # ηs = X * β
				obj = -loglikelihood(y,ηs)
				mul!(grad, GKGT ,β)
				obj += 0.5*λ*dot(grad,β)
				if obj < old_obj
					break
				else
					stephalving_count += 1
					@. β += t*direction
					t = t*0.5
				end
			end
			if stephalving_count==21
				@. β -= direction
			end
			total_iter = max_iters
		end
		βs[j,:] .= β
		println("λ=",λ, " sovled in ", total_iter ," iterations.")
	end
	return βs, time()-start_time
end


function kernel_logistic_fista(y::Vector, K::Matrix, G::Matrix, λ::Real;
                      verbose::Bool=true, tol::Real = 1e-4, max_iters::Int=3000)
	n = length(y)
	(r, s) = zeros(n), zeros(n) # work vectors
	m = size(G,1)
	β = zeros(m)
	KGT = K*Transpose(G)
	GKGT = G*KGT
	direction = similar(β)
	grad_res = similar(β)
	old_obj = Inf
	(γ, δ) = (copy(β), copy(β))
	nesterov = 0 
	grad = zeros(m)
	ηs = zeros(n)
	opnormKGT = opnorm(KGT)
	opnormGKGT = opnorm(GKGT)
	step_size = 1/(0.25*opnormKGT^2 + λ*opnormGKGT)
	t = 1
	objs = zeros(max_iters)
	grad_norms = zeros(max_iters)
	times = zeros(max_iters)
	start_time = time()
	for iter = 1:max_iters
		t_next = (1+sqrt(1+4*t^2))/2
		@. β = γ + ((t - 1)/t_next) * (γ - δ)
		@. δ = γ # Nesterov acceleration
		t = t_next
		mul!(ηs, KGT, β) # ηs = X * β
		for i = 1:n
			r[i] = y[i] - 1/(1+exp(-ηs[i]))
		end
		mul!(grad_res, Transpose(KGT), r) 
		@. grad_res *= -1
		obj = -loglikelihood(y,ηs)
		mul!(grad,GKGT,β)
		obj += 0.5*λ*dot(grad,β)
		@. grad *= λ
		@. grad += grad_res
		norm_grad = norm(grad, 2)
		if verbose
			println(iter,"  ",norm_grad,"  ", " ",obj, " ",nesterov)
		end
		objs[iter] = obj
		grad_norms[iter] = norm_grad
		times[iter] = time()-start_time
		@. β -= step_size*grad
		if norm_grad < tol && iter>1
			return (β, objs[1:iter], grad_norms[1:iter], times)
		else
			@. γ = β
			old_obj = obj
		end
	end
	return (β, objs, grad_norms, times)
end

function kernel_logistic_adagd(y::Vector, K::Matrix, G::Matrix,λ::Real;
                       verbose::Bool=true, tol::Real = 1e-4, max_iters::Int=3000)
	n = length(y)
	(r, s) = zeros(n), zeros(n) # work vectors
	β = zeros(m)
	KGT = K*Transpose(G)
	GKGT = G*KGT
	direction = similar(β)
	grad_res = similar(β)
	old_obj = Inf
	(β_prev, grad_prev) = (copy(β), copy(β))
	nesterov = 0 
	grad = zeros(m)
	ηs = zeros(n)
	objs = zeros(max_iters)
	grad_norms = zeros(max_iters)
	times = zeros(max_iters)
	start_time = time()
	θ = Inf
	t = t_prev = 1e-7
	for iter = 1:max_iters
		mul!(ηs, KGT, β) # ηs = X * β
		for i = 1:n
			r[i] = y[i] - 1/(1+exp(-ηs[i]))
		end
		mul!(grad_res, Transpose(KGT), r) 
		@. grad_res *= -1
		obj = -loglikelihood(y,ηs)
		mul!(grad,GKGT,β)
		obj += 0.5*λ*dot(grad,β)
		@. grad *= λ
		@. grad += grad_res
		norm_grad = norm(grad, 2)
		if verbose
			println(iter,"  ",norm_grad,"  ", " ",obj, " ",nesterov)
		end
		objs[iter] = obj
		grad_norms[iter] = norm_grad
		times[iter] = time()-start_time
		if norm_grad < tol 
			return (β, objs[1:iter], grad_norms[1:iter], times)
		end
		if iter==1
			@. β -= t*grad
			@. grad_prev = grad
		else
			t = min(sqrt(1+θ)*t_prev,norm(β-β_prev)/(2*norm(grad-grad_prev)))
			@. β_prev = β
			@. grad_prev = grad
			@. β -= t*grad
			θ = t/t_prev
			t_prev = t
		end
	end
	return (β, objs, grad_norms, times)
end