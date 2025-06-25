function grad_r!(s::Vector{T}, r::Vector{T}, τ::T, h::T, d) where T <: AbstractFloat
  for i in eachindex(r)
    s[i] = (τ-cdf(d,-r[i]))
  end
end

function lGu(u,d)
  (2/pi)^0.5*exp(-u^2/2)+u*(1-2*cdf(d,-u))
end

function QuantileLoss(r::Vector{T}, τ::T, h::T, d) where T <: AbstractFloat
  l = zero(T)
  taumhalf = τ-0.5
  hover2 = h/2
  for i in eachindex(r)
    l += hover2*lGu(r[i]/h,d)+taumhalf*r[i]
  end
  l
end

function FastQR(X::Matrix, y::Vector, τ::Real; h=0.25, tol=1e-4, verbose=true, kernel="uniform")
  ((n, p), T) = (size(X), eltype(X)) # cases, predictors, precision
  (h, max_iters) = (T(h), 5000)
  (r, s) = (zeros(T, n), zeros(T, n)) # work vectors
  XTy = Transpose(X) * y
  XTX = Transpose(X) * X
  L = cholesky!(XTX)
  β = L\XTy 
  inc = similar(β)
  grad = similar(β)
  old_obj = Inf
  (γ, δ) = (copy(β), copy(β))
  nesterov = 0 
  iter = 0
  if kernel == "uniform"
    d = Uniform(-h,h)
    c = 2*h
  else
    d = Normal(0,h)
    c = sqrt(2*pi)*h
  end
  d1 = Normal(0,1)
  nesterov = 0
  for iter = 1:max_iters
    nesterov = nesterov + 1
    @. β = γ + ((nesterov - 1)/(nesterov + 2)) * (γ - δ)
    @. δ = γ # Nesterov acceleration
    mul!(copy!(r, y), X, β, T(-1), T(1)) # r = y - X * β
    grad_r_linear!(s,r,τ,h,d)
    mul!(grad,Transpose(X),-s)
    norm_grad = norm(grad,2)
    obj = 1/n*QuantileLoss(r, τ, h, d1)
    if verbose
       println(iter,"  ",norm_grad,"  ", " ",obj, " ",nesterov)
       #println(iter, " ", norm_grad, " ",nesterov) 
    end
    @. XTy = c*n*grad
    #mul!(XTy, Transpose(X), r)
    ldiv!(inc, L, XTy)
    @. β -= inc
    if old_obj < obj
      nesterov = 0 
    end # restart Nesterov momentum
    if norm_grad < tol && iter>1
      return (β, norm_grad, iter, h)
    else
      @. γ = β
      old_obj = obj
    end
  end
  return (β,norm_grad, iter, h)
end



function grad_r_linear!(s::Vector{T},r::Vector{T}, τ::T, h::T, d) where T <: AbstractFloat
   oneovern = 1/length(s)
   for i in eachindex(r)
        s[i] = oneovern*(τ-cdf(d,-r[i]))
    end
end

# solve (MKϵ'Kϵ + ρKϵ)β = grad
function lin_solve_MM!(β::Vector, grad::Vector, M::Real, ρ::Real, ϵ::Real, e::Vector, U::Matrix, buffer::Vector)
  mul!(buffer,Transpose(U),grad)
  @. buffer /= M*(e+ϵ)^2 + ρ*(e+ϵ)
  mul!(β,U,buffer) 
end

function kernel_quantile(y::Vector, K::Matrix, G::AbstractMatrix, τ::Real, λ::Real;
                  h::Real = 0.25, verbose::Bool=true, tol::Real = 1e-4, 
                  ϵ::Real = 1e-9, adaptive=true)
  n = length(y)
  m = size(G,1)
  max_iters = 5000
  (r, s) = zeros(n), zeros(n) # work vectors
  β = zeros(m)
  KGT = K*Transpose(G)
  GKGT = G*KGT
  direction = similar(β)
  grad_res = similar(β)
  old_obj = Inf
  (γ, δ) = (copy(β), copy(β))
  d1 = Normal(0,1)
  d2 = Normal(0,h)
  grad = zeros(m)
  nesterov = 0
  H = 1/(sqrt(2*pi)*h)*Transpose(KGT)*KGT + λ*GKGT +ϵ*I
  L = cholesky!(H)
  objs = zeros(max_iters)
  grad_norms = zeros(max_iters)
  times = zeros(max_iters)
  start_time = time()
  for iter = 1:max_iters
    nesterov = nesterov + 1
    if adaptive
      @. β = γ + (nesterov /(nesterov + 2)) * (γ - δ)
    else
      @. β = γ + 0.333 * (γ - δ)
    end
    @. δ = γ # Nesterov acceleration
    mul!(copy!(r, y), KGT, β, -1.0, 1.0) # r = y - K * β
    grad_r!(s,r,τ,h,d2)
    s .*= -1
    mul!(grad_res, Transpose(KGT), s) 
    obj = QuantileLoss(r, τ, h, d1)
    mul!(grad,GKGT,β)
    obj += 0.5*λ*dot(grad,β)
    @. grad *= λ
    @. grad += grad_res
    norm_grad = norm(grad, 2)
    objs[iter] = obj
    grad_norms[iter] = norm_grad
    times[iter] = time()-start_time
    if norm_grad < tol && iter>1
      return (β,objs[1:iter], grad_norms[1:iter], times[1:iter])
    end
    if verbose
      println(iter,"  ",norm_grad,"  ", " ",obj, " ",nesterov)
    end
    #lin_solve_MM!(direction, grad, 1/(sqrt(2*pi)*h),λ, ϵ, e, U, s)
    ldiv!(direction,L,grad)
    @. β -= direction
    if old_obj < obj
      nesterov = 0 
    end # restart Nesterov momentum
     @. γ = β
     old_obj = obj
  end
  return (β,objs, grad_norms, times)
end

function kernel_quantile(y::Vector, K::Matrix, G::AbstractMatrix, τ::Real, λs::Vector;
                  h::Real = 0.25, verbose::Bool=true, tol::Real = 1e-4, 
                  ϵ::Real = 1e-9, adaptive=true)
  start_time = time()
  n = length(y)
  m = size(G,1)
  max_iters = 500
  (r, s) = zeros(n), zeros(n) # work vectors
  β = zeros(m)
  KGT = K*Transpose(G)
  GKGT = G*KGT
  direction = similar(β)
  grad_res = similar(β)
  old_obj = Inf
  (γ, δ) = (copy(β), copy(β))
  d1 = Normal(0,1)
  d2 = Normal(0,h)
  grad = zeros(m)
  nesterov = 0
  H_base = 1/(sqrt(2*pi)*h)*Transpose(KGT)*KGT+ϵ*I
  H = copy(H_base)
  βs = zeros(length(λs),m)
  for j in 1:length(λs)
    (γ, δ) = (copy(β), copy(β))
    total_iter = 0
    λ = λs[j]
    nesterov = 0
    old_obj = Inf
    @. H = H_base + λ*GKGT 
    L = cholesky!(H)
    for iter = 1:max_iters
      nesterov = nesterov + 1
      if adaptive
        @. β = γ + (nesterov /(nesterov + 2)) * (γ - δ)
      else
        @. β = γ + 0.333 * (γ - δ)
      end
      @. δ = γ # Nesterov acceleration
      mul!(copy!(r, y), KGT, β, -1.0, 1.0) # r = y - K * β
      grad_r!(s,r,τ,h,d2)
      s .*= -1
      mul!(grad_res, Transpose(KGT), s) 
      obj = QuantileLoss(r, τ, h, d1)
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
      #lin_solve_MM!(direction, grad, 1/(sqrt(2*pi)*h),λ, ϵ, e, U, s)
      ldiv!(direction,L,grad)
      @. β -= direction
      if old_obj < obj
        nesterov = 0 
      end # restart Nesterov momentum
      @. γ = β
      old_obj = obj
    end
    βs[j,:] .= β
    println("λ=",λ, " sovled in ", total_iter ," iterations.")
  end
  return βs, time()-start_time
end


function kernel_quantile_newton(y::Vector, K::Matrix, G::AbstractMatrix, τ::Real, λ::Real;
                      h::Real = 0.25, verbose::Bool=true, tol::Real = 1e-4, ϵ::Real = 1e-9)
  n = length(y)
  max_iters = 100
  m = size(G,1)
  (r, s) = zeros(n), zeros(n) # work vectors
  β = zeros(m)
  KGT = K*Transpose(G)
  GKGT = G*KGT
  direction = similar(β)
  grad_res = similar(β)
  old_obj = Inf
  (γ, δ) = (copy(β), copy(β))
  d1 = Normal(0,1)
  d2 = Normal(0,h)
  grad = zeros(m)
  objs = zeros(max_iters)
  grad_norms = zeros(max_iters)
  times = zeros(max_iters)
  start_time = time()
  H1 = zeros(n,m)
  H = zeros(m,m)
  ws = zeros(n)
  diagM = Diagonal(ones(m))
  for iter = 1:max_iters
    mul!(copy!(r, y), KGT, β, -1.0, 1.0) # r = y - K * β
    grad_r!(s,r,τ,h,d2)
    s .*= -1
    mul!(grad_res, Transpose(KGT), s) 
    old_obj = QuantileLoss(r, τ, h, d1)
    mul!(grad,GKGT,β)
    old_obj += 0.5*λ*dot(grad,β)
    @. grad *= λ
    @. grad += grad_res
    norm_grad = norm(grad, 2)
    objs[iter] = old_obj
    grad_norms[iter] = norm_grad
    times[iter] = time()-start_time
    if norm_grad < tol && iter>1
      return (β,objs[1:iter], grad_norms[1:iter], times[1:iter])
    end
    for i in 1:n
      ws[i] = pdf(d2, r[i])
    end
    mul!(H1,Diagonal(ws),KGT)
    mul!(H, Transpose(KGT),H1)
    @. H += λ*GKGT + ϵ*diagM
    direction .= H\grad
    t = 1
    stephalving_count = 0
    while stephalving_count<= 20
      @. β -= t*direction
      mul!(copy!(r, y), KGT, β, -1.0, 1.0) 
      obj = QuantileLoss(r, τ, h, d1)
      mul!(grad,GKGT,β)
      obj += 0.5*λ*dot(grad,β)
      if obj < old_obj
        break
      else
        stephalving_count += 1
        @. β += t*direction
        t = t*0.5
      end
    end
    if stephalving_count == 21
      @. β -= direction
    end
    if verbose
      println(iter,"  ",norm_grad,"  ", " ",old_obj, " ",stephalving_count)
    end
  end
  return (β,objs, grad_norms, times)
end

function kernel_quantile_newton(y::Vector, K::Matrix, G::AbstractMatrix, τ::Real, λs::Vector;
                       h::Real = 0.25, verbose::Bool=true, tol::Real = 1e-4, ϵ::Real = 1e-9)
  start_time = time()
  n = length(y)
  max_iters = 100
  (r, s) = zeros(n), zeros(n) # work vectors
  m = size(G,1)
  β = zeros(m)
  KGT = K*Transpose(G)
  GKGT = G*KGT
  direction = similar(β)
  grad_res = similar(β)
  old_obj = Inf
  (γ, δ) = (copy(β), copy(β))
  d1 = Normal(0,1)
  d2 = Normal(0,h)
  grad = zeros(m)
  H1 = zeros(n,m)
  H = zeros(m,m)
  ws = zeros(n)
  diagM = Diagonal(ones(m))
  βs = zeros(length(λs),m)
  for j in 1:length(λs)
    (obj, old_obj) = (0.0, Inf)
    total_iter = 0
    λ = λs[j]
    for iter = 1:max_iters
      mul!(copy!(r, y), KGT, β, -1.0, 1.0) # r = y - K * β
      grad_r!(s,r,τ,h,d2)
      s .*= -1
      mul!(grad_res, Transpose(KGT), s) 
      old_obj = QuantileLoss(r, τ, h, d1)
      mul!(grad,GKGT,β)
      old_obj += 0.5*λ*dot(grad,β)
      @. grad *= λ
      @. grad += grad_res
      norm_grad = norm(grad, 2)
      if norm_grad < tol && iter>1
        total_iter = iter
        break
      end
      for i in 1:n
        ws[i] = pdf(d2, r[i])
      end
      mul!(H1,Diagonal(ws),KGT)
      mul!(H, Transpose(KGT),H1)
      @. H += λ*GKGT + ϵ*diagM
      direction .= H\grad
      t = 1
      stephalving_count = 0
      while stephalving_count<= 20
        @. β -= t*direction
        mul!(copy!(r, y), KGT, β, -1.0, 1.0) 
        obj = QuantileLoss(r, τ, h, d1)
        mul!(grad,GKGT,β)
        obj += 0.5*λ*dot(grad,β)
        if obj < old_obj
          break
        else
          stephalving_count += 1
          @. β += t*direction
          t = t*0.5
        end
      end
      if stephalving_count == 21
        @. β -= direction
      end
      if verbose
        println(iter,"  ",norm_grad,"  ", " ",old_obj, " ",stephalving_count)
      end
    end
    βs[j,:] .= β
    println("λ=",λ, " sovled in ", total_iter ," iterations.")
  end
  return βs, time()-start_time
end


function kernel_quantile_fista(y::Vector, K::Matrix, G::Matrix, τ::Real, λ::Real;
                      h::Real=0.25, verbose::Bool=true, tol::Real = 1e-4)
  n = length(y)
  max_iters = 3000
  (r, s) = zeros(n), zeros(n) # work vectors
  m = size(G, 1)
  β = zeros(m)
  KGT = K*Transpose(G)
  GKGT = G*KGT
  direction = similar(β)
  grad_res = similar(β)
  old_obj = Inf
  (γ, δ) = (copy(β), copy(β))
  nesterov = 0 
  d1 = Normal(0,1)
  d2 = Normal(0,h)
  grad = zeros(m)
  nesterov = 0
  opnormGK = opnorm(KGT)
  opnormGKGT = opnorm(GKGT)
  step_size = 1/(1/(sqrt(2*pi)*h)*opnormGK^2 + λ*opnormGKGT)
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
    mul!(copy!(r, y), KGT, β, -1.0, 1.0) # r = y - K * β
    grad_r!(s,r,τ,h,d2)
    s .*= -1
    mul!(grad_res, Transpose(KGT), s) 
    obj = QuantileLoss(r, τ, h, d1)
    mul!(grad,GKGT,β)
    obj += 0.5*λ*dot(grad,β)
    @. grad *= λ
    @. grad += grad_res
    norm_grad = norm(grad, 2)
    objs[iter] = obj
    grad_norms[iter] = norm_grad
    times[iter] = time()-start_time
    if verbose
      println(iter,"  ",norm_grad,"  ", " ",obj, " ",nesterov)
    end
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

function kernel_quantile_adagd(y::Vector, K::Matrix, G::Matrix,τ::Real, λ::Real;
                      h::Real=0.25, verbose::Bool=true, tol::Real = 1e-4)
  n = length(y)
  max_iters = 3000
  (r, s) = zeros(n), zeros(n) # work vectors
  m = size(G,1)
  β = zeros(m)
  KGT = K*Transpose(G)
  GKGT = G*KGT
  grad = similar(β)
  grad_prev = similar(β)
  old_obj = Inf
  (β_prev, grad_prev) = (copy(β), copy(β))
  nesterov = 0 
  d1 = Normal(0,1)
  d2 = Normal(0,h)
  grad_res = similar(β)
  objs = zeros(max_iters)
  grad_norms = zeros(max_iters)
  times = zeros(max_iters)
  start_time = time()
  θ = Inf
  t = t_prev = 1e-7
  for iter = 1:max_iters
    mul!(copy!(r, y), KGT, β, -1.0, 1.0) # r = y - K * β
    grad_r!(s,r,τ,h,d2)
    s .*= -1
    mul!(grad_res, Transpose(KGT), s) 
    mul!(grad,GKGT,β)
    obj = QuantileLoss(r, τ, h, d1)
    obj += 0.5*λ*dot(grad,β)
    @. grad *= λ
    @. grad += grad_res
    norm_grad = norm(grad, 2)
    if verbose
      println(iter,"  ",norm_grad, " ", obj)
    end
    objs[iter] = obj
    grad_norms[iter] = norm_grad
    times[iter] = time()-start_time
    if norm_grad < tol
      return (β, objs[1:iter], grad_norms[1:iter], times[1:iter])
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