# QMME for kernel smoothed quantile regression (Gaussian smoothing, bandwidth h).
#   minimize  sum_i (h/2)*l(r_i/h) + (τ-1/2)*r_i + (λ/2)*β'Sβ,   r = y - Fβ,
# with F = K*G' (n-by-m) and S = G*K*G' (m-by-m), majorized by
#   H = F'F/(sqrt(2π)h) + λS + ϵI, refactored by Cholesky for each λ.
#
#   include("KernelQuantileQMME.jl")
#   βs, seconds, stats = KernelQuantileQMME.kernel_quantile(y, F, S, τ, λs)
#   β, objs, grad_norms, times = KernelQuantileQMME.kernel_quantile(y, F, S, τ, λ)
# Plain BLAS (set its threads with BLAS.set_num_threads), as the original.
module KernelQuantileQMME
using SpecialFunctions: erfc
include(joinpath(@__DIR__,"QMMEKernels.jl"))

# Loss at η = Fβ, and the score s = τ - Φ_h(-r) with r = y - η.
function quantile_loss!(s,eta,y,τ,h)
    loss=0.0; c=1/(h*sqrt(2.0)); a=sqrt(2/pi)
    @inbounds for i in eachindex(y)
        r=y[i]-eta[i]
        s[i]=τ-erfc(r*c)/2
        u=r/h
        loss+=h/2*(a*exp(-u^2/2)+u*(1-erfc(u/sqrt(2.0))))+(τ-0.5)*r
    end
    loss
end

"""Regularization path over λs with warm starts. Returns (βs, seconds, stats):
βs is length(λs)-by-m, seconds includes all preparation, stats has one entry
per λ (iterations, gradient, converged). Keywords: h, ϵ (damping of H only),
tol (default 1e-2), max_iters (default 500), grad_measure (:maxabs default, or :l2), adaptive / momentum
(:adaptive, :fixed, :none), restart_period (default 100), verbose.
"""
function kernel_quantile(y::Vector,F::Matrix,S::AbstractMatrix,τ::Real,λs::AbstractVector;
                         h=0.25,ϵ=1e-9,adaptive=true,momentum=adaptive ? :adaptive : :fixed,
                         history=false,kw...)
    start=time_ns()
    r=solve(y,F,Matrix{Float64}(S),Float64(τ),λs,h,ϵ,momentum,history;kw...)
    βs=permutedims(reduce(hcat,vec.(r.Bs)))
    history ? (βs,(time_ns()-start)*1e-9,r) : (βs,(time_ns()-start)*1e-9,r.stats)
end
"""Single λ: returns (β, objs, grad_norms, times); times exclude the factorization."""
function kernel_quantile(y::Vector,F::Matrix,S::AbstractMatrix,τ::Real,λ::Real;kw...)
    _,_,r=kernel_quantile(y,F,S,τ,[λ];history=true,kw...)
    (vec(r.Bs[1]),r.objs,r.grad_norms,r.times)
end

function solve(y,F,S,τ,λs,h,ϵ,momentum,history;kw...)
    n,m=size(F); length(y)==n || throw(DimensionMismatch("y and F"))
    c=1/(sqrt(2pi)*h)
    Hbase=c*(F'*F); Hbase+=ϵ*I
    H=similar(Hbase); cache=StableGramCache()
    fac=Ref{Cholesky{Float64,Matrix{Float64}}}()
    eta=zeros(n); s=zeros(n); pen=zeros(m)
    setup!(λ)=(@. H=Hbase+λ*S; fac[]=stable_gram_factor!(H,cache,F,S,c,λ,ϵ))
    function objective_gradient!(g,B,λ)
        β=vec(B); gv=vec(g)
        mul!(eta,F,β)
        obj=quantile_loss!(s,eta,y,τ,h)
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
