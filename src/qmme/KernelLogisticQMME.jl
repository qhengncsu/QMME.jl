# QMME for kernel logistic regression.
#   minimize  -sum_i [y_i η_i - log(1+exp(η_i))] + (λ/2)*β'Sβ,   η = Fβ,
# with F = K*G' (n-by-m) and S = G*K*G' (m-by-m), majorized by
#   H = F'F/4 + λS + ϵI, refactored by Cholesky for each λ.
#
#   include("KernelLogisticQMME.jl")
#   βs, seconds, stats = KernelLogisticQMME.kernel_logistic(y, F, S, λs)
#   β, objs, grad_norms, times = KernelLogisticQMME.kernel_logistic(y, F, S, λ)
# Plain BLAS (set its threads with BLAS.set_num_threads), as the original.
module KernelLogisticQMME
include(joinpath(@__DIR__,"QMMEKernels.jl"))

# Negative log-likelihood at η = Fβ (stable log(1+e^η)), and the residual
# s = y - p.
function logistic_loss!(s,eta,y)
    loss=0.0
    @inbounds for i in eachindex(y)
        η=eta[i]; yi=y[i]
        s[i]=yi-1/(1+exp(-η))
        loss-=yi*η-(max(η,0.0)+log1p(exp(-abs(η))))
    end
    loss
end

"""Regularization path over λs with warm starts. Returns (βs, seconds, stats):
βs is length(λs)-by-m, seconds includes all preparation, stats has one entry
per λ (iterations, gradient, converged). Keywords: ϵ (damping of H only),
tol (default 1e-2), max_iters (default 500), grad_measure (:maxabs default, or :l2), adaptive / momentum
(:adaptive, :fixed, :none), restart_period (default 100), verbose.
"""
function kernel_logistic(y::Vector,F::Matrix,S::AbstractMatrix,λs::AbstractVector;
                         ϵ=1e-9,adaptive=true,momentum=adaptive ? :adaptive : :fixed,
                         history=false,kw...)
    start=time_ns()
    r=solve(Vector{Float64}(y),F,Matrix{Float64}(S),λs,ϵ,momentum,history;kw...)
    βs=permutedims(reduce(hcat,vec.(r.Bs)))
    history ? (βs,(time_ns()-start)*1e-9,r) : (βs,(time_ns()-start)*1e-9,r.stats)
end
"""Single λ: returns (β, objs, grad_norms, times); times exclude the factorization."""
function kernel_logistic(y::Vector,F::Matrix,S::AbstractMatrix,λ::Real;kw...)
    _,_,r=kernel_logistic(y,F,S,[λ];history=true,kw...)
    (vec(r.Bs[1]),r.objs,r.grad_norms,r.times)
end

function solve(y,F,S,λs,ϵ,momentum,history;kw...)
    n,m=size(F); length(y)==n || throw(DimensionMismatch("y and F"))
    c=0.25
    Hbase=c*(F'*F); Hbase+=ϵ*I
    H=similar(Hbase); cache=StableGramCache()
    fac=Ref{Cholesky{Float64,Matrix{Float64}}}()
    eta=zeros(n); s=zeros(n); pen=zeros(m)
    setup!(λ)=(@. H=Hbase+λ*S; fac[]=stable_gram_factor!(H,cache,F,S,c,λ,ϵ))
    function objective_gradient!(g,B,λ)
        β=vec(B); gv=vec(g)
        mul!(eta,F,β)
        obj=logistic_loss!(s,eta,y)
        mul!(gv,F',s)
        mul!(pen,S,β)
        @. gv=-gv+λ*pen
        obj+0.5*λ*dot(β,pen)
    end
    direction!(d,g,λ)=ldiv!(d,fac[],g)
    qmme_path(zeros(m,1),λs,setup!,objective_gradient!,direction!;
              momentum=momentum,history=history,kw...)
end
end
