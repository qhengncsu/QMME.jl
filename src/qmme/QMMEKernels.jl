# Shared building blocks for the QMME method files (KernelQuantileQMME.jl,
# KernelLogisticQMME.jl, KernelMultinomialAlg2.jl, KernelMultinomialAlg3.jl).
# Each method file includes this file inside its own module.
#
# Plain Julia, no packages, no Accelerate: runs on Linux, macOS and Windows,
# x86-64 and aarch64. Quantile and logistic regression use BLAS only (the QMME
# loop and the stable Gram factor below). Multinomial regression also uses the
# fused, threaded pass over F with two hand-written SIMD kernels; start
# Julia with `julia -t N` to use N threads.
using LinearAlgebra, Base.Cartesian, Base.Threads

# ---------------------------------------------------------------- SIMD
# <2 x double> is a NEON register on aarch64 and an SSE/AVX register on x86.
# The IR uses no pointers, so it is unaffected by LLVM's opaque-pointer change.
# `contract` lets LLVM fuse the multiply and add into one FMA instruction.
const V2=NTuple{2,VecElement{Float64}}
const ZERO=(VecElement(0.0),VecElement(0.0))
@inline vfma(a::V2,b::V2,c::V2)=Base.llvmcall(
    "%a = fmul contract <2 x double> %0, %1\n%r = fadd contract <2 x double> %a, %2\nret <2 x double> %r",
    V2,Tuple{V2,V2,V2},a,b,c)
# unsafe_load uses alignment 1, so unaligned addresses are safe on every CPU.
@inline vload(p::Ptr{Float64})=unsafe_load(Ptr{V2}(p))
@inline vbcast(x::Float64)=(VecElement(x),VecElement(x))
@inline vsum(a::V2)=a[1].value+a[2].value

# out[r0+w, j0+j] = sum_k At[k, c0+w]*X[k, j0+j] for w=1:len, j=1:D.
# Four columns of At share each load of X; k runs in SIMD pairs.
@generated function tmul_block!(out,r0,At,c0,len,X,j0,::Val{D}) where D
    W=4
    quote
        m=size(At,1); sA=m; sX=size(X,1); w=0
        pA=pointer(At); pX=pointer(X)
        GC.@preserve At X begin
        @inbounds while w+$W<=len
            @nexprs $W c->(@nexprs $D j->(a_c_j=ZERO))
            k=0
            while k+2<=m
                @nexprs $D j->(x_j=vload(pX+8*(k+(j0+j-1)*sX)))
                @nexprs $W c->(f=vload(pA+8*(k+(c0+w+c-1)*sA));
                               @nexprs $D j->(a_c_j=vfma(f,x_j,a_c_j)))
                k+=2
            end
            @nexprs $W c->(@nexprs $D j->begin
                s=vsum(a_c_j)
                for kk=k+1:m; s=muladd(At[kk,c0+w+c],X[kk,j0+j],s); end
                out[r0+w+c,j0+j]=s
            end)
            w+=$W
        end
        @inbounds while w<len
            for j=1:$D
                s=0.0
                for k=1:m; s=muladd(At[k,c0+w+1],X[k,j0+j],s); end
                out[r0+w+1,j0+j]=s
            end
            w+=1
        end
        end
        out
    end
end
# g[k, j0+j] += sum_t At[k, c0+t]*r[t, j0+j]; four contiguous k per pass.
@generated function tmulT_block!(g,At,c0,len,r,j0,::Val{D}) where D
    quote
        m=size(At,1); sA=m; k0=0; pA=pointer(At)
        GC.@preserve At begin
        @inbounds while k0+4<=m
            @nexprs 2 c->(@nexprs $D j->(a_c_j=ZERO))
            for t=1:len
                @nexprs $D j->(r_j=vbcast(r[t,j0+j]))
                p=pA+8*(k0+(c0+t-1)*sA)
                @nexprs 2 c->(f_c=vload(p+16*(c-1)))
                @nexprs 2 c->(@nexprs $D j->(a_c_j=vfma(f_c,r_j,a_c_j)))
            end
            @nexprs 2 c->(@nexprs $D j->(g[k0+2c-1,j0+j]+=a_c_j[1].value;
                                         g[k0+2c,j0+j]+=a_c_j[2].value))
            k0+=4
        end
        @inbounds for k=k0+1:m, j=1:$D
            s=0.0
            for t=1:len; s=muladd(At[k,c0+t],r[t,j0+j],s); end
            g[k,j0+j]+=s
        end
        end
        g
    end
end

# Widest column chunk whose accumulators fit in registers: tmul_block! holds
# 4D accumulators plus D+1 operands (5D+1 vector registers), so D<=6 with the
# 32 registers of aarch64 and D<=3 with the 16 of x86-64 (AVX2).
# Override with the environment variable QMME_MAX_COLS or the max_cols keyword.
function default_max_cols()
    v=get(ENV,"QMME_MAX_COLS","")
    isempty(v) || return clamp(parse(Int,v),1,6)
    Sys.ARCH===:aarch64 ? 6 : 3
end
const MAX_COLS=Ref(6)
__init__()=(MAX_COLS[]=default_max_cols())

# Split d columns into balanced chunks of at most maxc (7 -> 4+3, not 6+1).
@inline function chunked(f,d,maxc)
    j0=0; left=cld(d,maxc)
    while j0<d
        c=cld(d-j0,left); left-=1
        c==6 ? f(j0,Val(6)) : c==5 ? f(j0,Val(5)) : c==4 ? f(j0,Val(4)) :
        c==3 ? f(j0,Val(3)) : c==2 ? f(j0,Val(2)) : f(j0,Val(1))
        j0+=c
    end
end

const ROWS=64   # rows of F per task: the block stays in cache between both products

# out = At' * X, threaded over blocks of output rows.
function tmul!(out,At,X,maxc)
    p=size(At,2); d=size(X,2)
    @sync for b=1:cld(p,ROWS)
        @spawn begin
            c0=(b-1)*ROWS; len=min(ROWS,p-c0)
            chunked((j0,D)->tmul_block!(out,c0,At,c0,len,X,j0,D),d,maxc)
        end
    end
    out
end

# ---------------------------------------------------- fused F pass
# One pass over F (stored transposed): eta=F*B, then block_loss!(R,eta,r0,len)
# returns the loss of rows r0+1:r0+len and writes their residual into R, then
# G+=F'*R. Each row block of F is read from memory once per iteration. Both
# products are SIMD kernels, so no BLAS is called inside the threaded region.
struct Workspace
    eta::Vector{Matrix{Float64}}
    R::Vector{Matrix{Float64}}
    G::Vector{Matrix{Float64}}
    losses::Vector{Float64}
end
function Workspace(n,m,d)
    nt=nthreads()
    Workspace([zeros(ROWS,d) for _=1:nt],[zeros(ROWS,d) for _=1:nt],
              [zeros(m,d) for _=1:nt],zeros(cld(n,ROWS)))
end
function loss_gradient!(G,ws,Ft,B,block_loss!,maxc)
    m,n=size(Ft); d=size(B,2); nblk=cld(n,ROWS)
    next=Atomic{Int}(1); ntask=min(length(ws.G),nblk)
    @sync for t=1:ntask
        @spawn begin
            g=ws.G[t]; e=ws.eta[t]; r=ws.R[t]; fill!(g,0.0)
            while true
                b=atomic_add!(next,1); b>nblk && break
                c0=(b-1)*ROWS; len=min(ROWS,n-c0)
                chunked((j0,D)->tmul_block!(e,0,Ft,c0,len,B,j0,D),d,maxc)
                ws.losses[b]=block_loss!(r,e,c0,len)
                chunked((j0,D)->tmulT_block!(g,Ft,c0,len,r,j0,D),d,maxc)
            end
        end
    end
    copyto!(G,ws.G[1])
    for t=2:ntask; G.+=ws.G[t]; end
    sum(ws.losses)
end

# ------------------------------------------------ multinomial helpers
function class_labels(Y)
    n,q=size(Y); labels=Vector{Int}(undef,n)
    for i=1:n
        k=argmax(@view Y[i,:]); labels[i]=k
        all(j->Y[i,j]==(j==k ? 1 : 0),1:q) || throw(ArgumentError("Y must be one-hot"))
    end
    labels
end
# Stable softmax for one block of rows: negative log-likelihood, and the
# residual Y-P in R. full=false: the last class is the reference (eta=0).
function multinomial_block!(R,eta,labels,full,r0,len)
    d=size(eta,2); loss=0.0
    @inbounds for i=1:len
        mx=full ? -Inf : 0.0
        for j=1:d; mx=max(mx,eta[i,j]); end
        total=full ? 0.0 : exp(-mx)
        for j=1:d
            v=exp(eta[i,j]-mx); R[i,j]=v; total+=v
        end
        target=labels[r0+i]
        chosen=target<=d ? eta[i,target] : 0.0
        loss-=(chosen-mx)-log(total)
        for j=1:d; R[i,j]=(j==target ? 1.0 : 0.0)-R[i,j]/total; end
    end
    loss
end

# ------------------------------------------------------ factorizations
# Cholesky factorizations and triangular solves use BLAS/LAPACK directly.
# Symmetric eigendecomposition; divide and conquer when LAPACK.syevd! exists.
function sym_eigen(A::Matrix{Float64})
    A=(A+A')/2
    isdefined(LAPACK,:syevd!) && return LAPACK.syevd!('V','U',A)
    e=eigen(Symmetric(A)); (e.values,e.vectors)
end

# Cholesky factor of H = c*F'F + λ*S + ϵI without changing the damping
# (as figures345_20reps/source/StableGram.jl). Forming c*F'F can lose small
# eigenvalues when the columns of F are nearly dependent; if the factor is
# ill-conditioned, QR of the augmented design [sqrt(c)F; sqrt(ϵ)I; sqrt(λ)R_S]
# gives a factor of the same matrix stably. H is overwritten.
mutable struct StableGramCache
    base::Union{Nothing,Matrix{Float64}}
    penalty::Union{Nothing,Matrix{Float64}}
end
StableGramCache()=StableGramCache(nothing,nothing)
function stable_gram_factor!(H,cache::StableGramCache,F,S,c,λ,ϵ;verbose=false)
    fac=cholesky!(Symmetric(H,:U);check=false)
    rcond=issuccess(fac) ? LAPACK.trcon!('1','U','N',fac.factors) : 0.0
    rcond>=sqrt(eps(Float64)) && return fac
    if cache.base===nothing
        m=size(F,2)
        cache.base=Matrix(qr!(vcat(sqrt(c)*F,sqrt(ϵ)*Matrix{Float64}(I,m,m))).R)
        cache.penalty=Matrix(cholesky(Symmetric(S)).U)
    end
    verbose && println("STABLE_GRAM_QR lambda=",λ," rcond=",rcond)
    Cholesky(Matrix(qr!(vcat(cache.base,sqrt(λ)*cache.penalty)).R),'U',0)
end

# ----------------------------------------------------- QMME iteration
# The common path loop. For each λ: setup!(λ); then from the warm start,
#   y = x + beta*(x - x_prev);  obj = objective_gradient!(grad, y, λ)
#   stop if the gradient at y passes, or at the iteration cap (return y);
#   otherwise x = y - direction!(dir, grad, λ).
# momentum: :adaptive (l/(l+2)), :fixed (0.333) or :none (beta=0).
# Restarts (Algorithm 1): the counter l is reset to 1 when f(y) increases
# over the previous anchor or when l would reach restart_period = P, so
# l ∈ {1,...,P-1} and beta ≤ (P-1)/(P+1).
# Defaults: stop when max_i |grad_i| < 1e-2, at most 500 iterations per λ
# (pass max_iters=1000 explicitly for the final refit).
function qmme_path(B0::Matrix{Float64},λs,setup!,objective_gradient!,direction!;
                   tol=1e-2,max_iters=500,grad_measure=:maxabs,momentum=:adaptive,
                   restart_period=100,verbose=false,history=false)
    max_iters>0 || throw(ArgumentError("max_iters must be positive"))
    restart_period>1 || throw(ArgumentError("restart_period must be at least 2"))
    all(x->isfinite(x)&&x>0,λs) || throw(ArgumentError("positive finite lambdas required"))
    grad_measure in (:l2,:maxabs) || throw(ArgumentError("unknown gradient measure"))
    momentum in (:adaptive,:fixed,:none) || throw(ArgumentError("unknown momentum"))
    B=copy(B0); x=copy(B0); xprev=copy(B0); grad=similar(B); dir=similar(B)
    Bs=Matrix{Float64}[]; stats=NamedTuple[]
    objs=Float64[]; norms=Float64[]; times=Float64[]
    start=time_ns()
    for (j,λ) in enumerate(λs)
        setup!(λ)
        j==1 && (start=time_ns())   # history times exclude the first factorization
        copyto!(x,B); copyto!(xprev,B)
        old_obj=Inf; l=0; ng=Inf; iters=0; restarts=0
        for iter=1:max_iters
            l+=1
            beta=momentum===:adaptive ? l/(l+2) : momentum===:fixed ? 0.333 : 0.0
            @. B=x+beta*(x-xprev)
            copyto!(xprev,x)
            obj=objective_gradient!(grad,B,λ)
            ng=grad_measure===:l2 ? norm(grad) : maximum(abs,grad)
            isfinite(ng) || error("Nonfinite gradient at lambda=$λ")
            if history
                push!(objs,obj);push!(norms,ng);push!(times,(time_ns()-start)*1e-9)
            end
            verbose && println(iter," ",ng," ",obj," ",l)
            iters=iter
            (ng<tol || iter==max_iters) && break
            direction!(dir,grad,λ)
            @. x=B-dir
            copyto!(B,x)
            old_obj<obj && (l=0;restarts+=1)
            l+1>=restart_period && (l=0)
            old_obj=obj
        end
        push!(Bs,copy(B))
        push!(stats,(lambda=λ,iterations=iters,gradient=ng,converged=ng<tol,restarts=restarts))
    end
    (Bs=Bs,stats=stats,objs=objs,grad_norms=norms,times=times)
end
