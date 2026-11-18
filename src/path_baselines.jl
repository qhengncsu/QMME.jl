# Independent Newton-PCG baseline: block Jacobi of the current true Hessian.
# Contiguous coefficient blocks within each class; no QMME majorizer is used.
function hvp(problem,F,S,λ,w,V; ridge=default_ridge(problem))
    T=F*V
    if problem=="multinomial"
        T=w.*(T.-sum(w.*T;dims=2))
    else
        T .*= w
    end
    F'*T + λ*(S*V) + (problem=="multinomial" ? λ*ridge : ridge)*V
end
function block_jacobi(problem,F,S,λ,w; blocksize=64,ridge=default_ridge(problem))
    m=size(F,2);d=size(w,2); blocks=[]
    for k=1:d, lo=1:blocksize:m
        ix=lo:min(lo+blocksize-1,m); Z=F[:,ix]
        c=problem=="multinomial" ? w[:,k].*(1 .- w[:,k]) : vec(w)
        D=Z'*(c.*Z)+λ*S[ix,ix]+(problem=="multinomial" ? λ*ridge : ridge)*I
        push!(blocks,((k-1)*m .+ ix,cholesky(Symmetric(D))))
    end
    blocks
end
function pcg_direction(problem,F,S,λ,w,g,blocks; forcing=.1,maxiter=2000)
    apply(r)=begin
        z=similar(r)
        for (ix,C) in blocks; z[ix]=C\r[ix]; end
        z
    end
    x=zeros(length(g));r=vec(copy(g));z=apply(r);p=copy(z);rz=dot(r,z);ng=norm(r)
    for it=1:maxiter
        Ap=vec(hvp(problem,F,S,λ,w,reshape(p,size(g)))); curvature=dot(p,Ap)
        curvature>0 || return reshape(x,size(g)),it
        α=rz/curvature;x .+= α.*p;r .-= α.*Ap
        if norm(r)<=forcing*ng
            r .= vec(g) .- vec(hvp(problem,F,S,λ,w,reshape(x,size(g))))
            norm(r)<=forcing*ng && return reshape(x,size(g)),it
            z=apply(r);p=copy(z);rz=dot(r,z);continue
        end
        z=apply(r);next=dot(r,z);p .= z .+ (next/rz).*p;rz=next
    end
    reshape(x,size(g)),maxiter
end
function competitive_path(problem,method,F,S,y,λs;tol=1e-4,max_iters=1000,memory=5,time_limit=3600.)
    B=zeros(size(F,2),nvars(problem,y));Bs=Matrix{Float64}[];stats=NamedTuple[];start=time()
    for λ in λs
        iters=0;cgits=0;status="iteration_limit"
        if time()-start>=time_limit
            status="time_limit"
        elseif method=="LBFGS"
            function fg!(f,g,x)
                val,gr,_=objective(problem,F,S,y,λ,reshape(x,size(B));gradient=g!==nothing)
                g===nothing || copyto!(g,gr)
                f===nothing ? nothing : val
            end
            cb=s->norm(s.g_x)<tol || time()-start>=time_limit
            opts=Optim.Options(iterations=max_iters,g_abstol=0.,f_abstol=0.,f_reltol=0.,x_abstol=0.,x_reltol=0.,callback=cb)
            r=Optim.optimize(Optim.NLSolversBase.only_fg!(fg!),vec(copy(B)),Optim.LBFGS(m=memory),opts)
            B .= reshape(Optim.minimizer(r),size(B));iters=Optim.iterations(r)
            iters<max_iters && (status="optimizer_stopped")
        else
            for it=1:max_iters
                f,g,w=objective(problem,F,S,y,λ,B);gn=norm(g);iters=it-1
                gn<tol && break
                if time()-start>=time_limit;status="time_limit";break;end
                blocks=block_jacobi(problem,F,S,λ,w)
                dir,ci=pcg_direction(problem,F,S,λ,w,g,blocks;forcing=min(.1,sqrt(gn)),maxiter=2000)
                cgits+=ci; dot(g,dir)>0 || (dir=copy(g))
                t=1.;accepted=false
                for ls=1:40
                    candidate=B.-t.*dir
                    fc=objective(problem,F,S,y,λ,candidate;gradient=false)[1]
                    if fc<=f-1e-4*t*dot(g,dir)
                        B .= candidate;accepted=true;break
                    end
                    t/=2
                end
                if !accepted;status="line_search_failed";break;end
                iters=it
            end
        end
        f,g,_=objective(problem,F,S,y,λ,B);ng=norm(g)
        ng<tol && (status="converged")
        time()-start>=time_limit && ng>=tol && (status="time_limit")
        get(ENV,"QMME_PROGRESS","0")=="1" && (@printf("  %s %s lambda=%.3g iterations=%d grad=%.3g status=%s elapsed=%.1f\n",problem,method,λ,iters,ng,status,time()-start);flush(stdout))
        push!(Bs,copy(B));push!(stats,(lambda=λ,iterations=iters,gradient=ng,converged=ng<tol,cg_iterations=cgits,status=status))
    end
    Bs,stats
end
