# QMME for kernel multinomial regression, Algorithm 3 (variable transformation).
#   minimize  -loglik(Y; FB) + (λ/2)*tr(B'SB),
# with F = K*G' (n-by-m), S = G*K*G' (m-by-m); B is m-by-(q-1) (standard,
# last class is the reference) or m-by-q (full). Direction solve: whiten by
# S+ϵI = LL', eigendecompose L^{-1}F'FL^{-T} once, and use the two-eigenvalue
# structure of E; all decompositions are reused across λ.
#
#   include("KernelMultinomialAlg3.jl")
#   Bs, seconds, stats = KernelMultinomialAlg3.KernelMultinomial(Y, F, S, λs)       # Bs: L×m×(q-1)
#   B, objs, grad_norms, times = KernelMultinomialAlg3.KernelMultinomial(Y, F, S, λ)
#   KernelMultinomialAlg3.KernelMultinomial_full(...)   # full parameterization
# Start Julia with `-t N` to use N threads.
module KernelMultinomialAlg3
include(joinpath(@__DIR__,"QMMEKernels.jl"))

# Class step for E = (I - 11'/(d+1))/2: eigenvalue 1/(2(d+1)) on the mean
# direction and 1/2 on its complement.
function class_solve!(out,rhs,means,den_center,den_mean)
    m,d=size(rhs); fill!(means,0.0)
    @inbounds for j=1:d, i=1:m; means[i]+=rhs[i,j]; end
    @inbounds for i=1:m; means[i]/=d; end
    @inbounds for j=1:d, i=1:m
        out[i,j]=(rhs[i,j]-means[i])/den_center[i]+means[i]/den_mean[i]
    end
    out
end

"""Regularization path over λs with warm starts. Returns (Bs, seconds, stats):
Bs is length(λs)-by-m-by-d, seconds includes all preparation, stats has one
entry per λ. Keywords: ϵ (damping of S in the majorizer), tol (default 1e-2),
max_iters (default 500), grad_measure (:maxabs default, or :l2), adaptive /
momentum (:adaptive, :fixed, :none),
restart_period (default 100), Binit, verbose, max_cols.
"""
KernelMultinomial(Y::Matrix,F::Matrix,S::AbstractMatrix,λs::AbstractVector;kw...)=path(Y,F,S,λs,false;kw...)
KernelMultinomial_full(Y::Matrix,F::Matrix,S::AbstractMatrix,λs::AbstractVector;kw...)=path(Y,F,S,λs,true;kw...)
"""Single λ: returns (B, objs, grad_norms, times); times exclude the factorization."""
KernelMultinomial(Y::Matrix,F::Matrix,S::AbstractMatrix,λ::Real;kw...)=single(Y,F,S,λ,false;kw...)
KernelMultinomial_full(Y::Matrix,F::Matrix,S::AbstractMatrix,λ::Real;kw...)=single(Y,F,S,λ,true;kw...)

function single(Y,F,S,λ,full;kw...)
    _,_,r=path(Y,F,S,[λ],full;history=true,kw...)
    (r.Bs[1],r.objs,r.grad_norms,r.times)
end
function path(Y,F,S,λs,full;ϵ=1e-4,adaptive=true,momentum=adaptive ? :adaptive : :fixed,
              max_cols=MAX_COLS[],history=false,Binit=nothing,kw...)
    start=time_ns()
    r=solve(Y,F,Matrix{Float64}(S),λs,full,ϵ,momentum,max_cols,history,Binit;kw...)
    Bs=permutedims(cat(r.Bs...;dims=3),(3,1,2))
    history ? (Bs,(time_ns()-start)*1e-9,r) : (Bs,(time_ns()-start)*1e-9,r.stats)
end

function solve(Y,F,S,λs,full,ϵ,momentum,maxc,history,Binit;kw...)
    n,m=size(F); q=size(Y,2); d=full ? q : q-1
    size(Y,1)==n && d>0 || throw(DimensionMismatch("Y and F"))
    labels=class_labels(Y)
    # Whitening and spectral decomposition, reused for every λ.
    L=cholesky(Symmetric(S+ϵ*I)).L
    Fw=F/L'
    a,Γ=sym_eigen(Fw'*Fw)
    minimum(a)>=-1e-10*max(1.,maximum(abs,a)) || error("Invalid whitened curvature")
    a=max.(a,0.)
    J=L'\Γ; Jt=permutedims(J); Ft=permutedims(F); St=permutedims(S)
    ws=Workspace(n,m,d); pen=zeros(m,d); rhs=zeros(m,d); scaled=zeros(m,d)
    means=zeros(m); dc=zeros(m); dm=zeros(m)
    block=(R,e,r0,len)->multinomial_block!(R,e,labels,full,r0,len)
    setup!(λ)=(@. dc=a/2+λ; @. dm=a/(2*(d+1))+λ)
    function objective_gradient!(g,B,λ)
        obj=loss_gradient!(g,ws,Ft,B,block,maxc)
        tmul!(pen,St,B,maxc)
        @. g=-g+λ*pen
        obj+0.5*λ*dot(B,pen)
    end
    function direction!(dir,g,λ)
        tmul!(rhs,J,g,maxc)                 # J'g
        class_solve!(scaled,rhs,means,dc,dm)
        tmul!(dir,Jt,scaled,maxc)           # J*scaled
    end
    B0=Binit===nothing ? zeros(m,d) : Matrix{Float64}(Binit)
    size(B0)==(m,d) || throw(DimensionMismatch("Binit"))
    qmme_path(B0,λs,setup!,objective_gradient!,direction!;
              momentum=momentum,history=history,kw...)
end
end
