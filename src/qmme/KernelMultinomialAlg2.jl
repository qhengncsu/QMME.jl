# QMME for kernel multinomial regression, Algorithm 2 (Schur decomposition and
# forward substitution).
#   minimize  -loglik(Y; FB) + (λ/2)*tr(B'SB),
# with F = K*G' (n-by-m), S = G*K*G' (m-by-m); B is m-by-(q-1) (standard,
# last class is the reference) or m-by-q (full). Direction solve: Cholesky of
# S+ϵI, Schur decomposition of (S+ϵI)^{-1}F'F and spectral decomposition of
# E^{-1} = 2(I+11'), computed once and reused across λ; the shifted
# triangular systems are solved with BLAS, one shift per eigenvalue of E^{-1}.
#
#   include("KernelMultinomialAlg2.jl")
#   Bs, seconds, stats = KernelMultinomialAlg2.KernelMultinomial(Y, F, S, λs)       # Bs: L×m×(q-1)
#   B, objs, grad_norms, times = KernelMultinomialAlg2.KernelMultinomial(Y, F, S, λ)
#   KernelMultinomialAlg2.KernelMultinomial_full(...)   # full parameterization
# Start Julia with `-t N` to use N threads.
module KernelMultinomialAlg2
include(joinpath(@__DIR__,"QMMEKernels.jl"))

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
    # Decompositions reused for every λ (as in the original Algorithm 2).
    E=2 .*(Matrix{Float64}(I,d,d)+ones(d,d))
    e2,V=eigen(Symmetric(E))
    Sc=cholesky(Symmetric(S+ϵ*I))
    sch=schur(transpose(Sc\(F'*F)))
    Aprime=Matrix(transpose(sch.T))          # lower triangular
    U=sch.Z; Ut=permutedims(U); Ft=permutedims(F); St=permutedims(S)
    # Columns of the transformed system share a diagonal shift λ*e2[l]; E^{-1}
    # has two distinct eigenvalues, so group the columns by shift.
    groups=Vector{Vector{Int}}()
    for l=1:d
        k=findfirst(gr->isapprox(e2[gr[1]],e2[l];rtol=1e-10),groups)
        k===nothing ? push!(groups,[l]) : push!(groups[k],l)
    end
    T=[similar(Aprime) for _ in groups]; Xg=[zeros(m,length(gr)) for gr in groups]
    ws=Workspace(n,m,d); pen=zeros(m,d); rhs=zeros(m,d); b1=zeros(m,d); b2=zeros(m,d)
    block=(R,e,r0,len)->multinomial_block!(R,e,labels,full,r0,len)
    function setup!(λ)
        for (k,gr) in enumerate(groups)
            copyto!(T[k],Aprime)
            for i=1:m; T[k][i,i]+=λ*e2[gr[1]]; end
        end
    end
    function objective_gradient!(g,B,λ)
        obj=loss_gradient!(g,ws,Ft,B,block,maxc)
        tmul!(pen,St,B,maxc)
        @. g=-g+λ*pen
        obj+0.5*λ*dot(B,pen)
    end
    function direction!(dir,g,λ)
        ldiv!(rhs,Sc,g)                      # (S+ϵI)^{-1} g
        mul!(b2,rhs,E)
        tmul!(b1,U,b2,maxc)                  # U' * ...
        mul!(b2,b1,V)
        for (k,gr) in enumerate(groups)      # (A' + λ e2_l I) x_l = b_l
            @views Xg[k].=b2[:,gr]
            ldiv!(LowerTriangular(T[k]),Xg[k])
            @views b2[:,gr].=Xg[k]
        end
        tmul!(b1,Ut,b2,maxc)                 # U * ...
        mul!(dir,b1,V')
    end
    B0=Binit===nothing ? zeros(m,d) : Matrix{Float64}(Binit)
    size(B0)==(m,d) || throw(DimensionMismatch("Binit"))
    qmme_path(B0,λs,setup!,objective_gradient!,direction!;
              momentum=momentum,history=history,kw...)
end
end
