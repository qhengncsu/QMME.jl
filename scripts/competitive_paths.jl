using QMMEReproducibility,LinearAlgebra,Statistics,Printf
const Q=QMMEReproducibility
# Store each completed path separately so runs can resume between methods.
const OUT = mkpath(get(ENV, "QMME_OUT", resultdir("competitive_paths")))
const REPS=parse(Int,get(ENV,"QMME_REPS","3"));const MS=parse.(Int,split(get(ENV,"QMME_MS","128,512"),','))
const N=parse(Int,get(ENV,"QMME_N","16384"));const CAP=parse(Float64,get(ENV,"QMME_CAP","3600"))
const METHODS=String.(split(get(ENV,"QMME_METHODS","QMME,Newton,LBFGS"),','))
all(in(("QMME", "Newton", "LBFGS", "Newton-PCG")), METHODS) || error("Unknown path method")
# Compile all paths before measuring any timings.
for p in ("quantile","logistic","multinomial")
 Q.warmup_sim(p);F,S,ys=Q.trajectory_instance(100,10;σ=5.)
 for md in filter(x->x in ("LBFGS","Newton-PCG"),METHODS); Q.competitive_path(p,md,F,S,ys[p],[1.,.1]);end
end
for rep=1:REPS
 inst=Q.sim_instance(rep,1;n=N);σ=Q.median_distance(inst.X)
 for m in MS
  t=time();F=Q.rbf(inst.X,inst.X[inst.idx[1:m],:],σ);S=Matrix(Symmetric(F[inst.idx[1:m],:]));kernel_seconds=time()-t
  Fv=Q.rbf(inst.Xv,inst.X[inst.idx[1:m],:],σ)
  for p in ("quantile","logistic","multinomial")
   y=p=="quantile" ? inst.yq : p=="logistic" ? inst.yl : Q.sim_labels(inst.a,inst.b,3,inst.seed+30)
   yv=p=="quantile" ? inst.av : p=="logistic" ? inst.ylv : Q.sim_labels(inst.av,inst.bv,3,inst.seed+31)
   for method in circshift(METHODS,rep-1)
    file=joinpath(OUT,"$(p)_m$(m)_rep$(rep)_$(method).csv");isfile(file)&&continue
    GC.gc();t=time()
    if method=="LBFGS"
     Bs,stats=Q.competitive_path(p,"LBFGS",F,S,y,Q.LAMBDAS;memory=5,max_iters=1000,time_limit=CAP)
    elseif method=="Newton-PCG"
     Bs,stats=Q.competitive_path(p,method,F,S,y,Q.LAMBDAS;max_iters=1000,time_limit=CAP)
    else
     Bs,stats=Q.fit_path(p,method=="QMME"&&p=="multinomial" ? "Alg3-standard" : method,F,S,y)
    end
    secs=time()-t;ng=[norm(Q.objective(p,F,S,y,λ,B)[2]) for (λ,B) in zip(Q.LAMBDAS,Bs)]
    val=[Q.validation(p,B,Fv,yv) for B in Bs]
    tmp = file * ".tmp" * string(getpid())
    open(tmp,"w") do io
     println(io,"problem,n,m,rep,seed,sigma,method,lambda,solver_seconds,kernel_seconds,gradient,objective,validation,iterations,cg_iterations,status")
     for j in eachindex(Bs)
      st=stats[j];status=ng[j]<1e-4 ? "converged" : haskey(st,:status) ? st.status : "iteration_limit"
      println(io,join((p,N,m,rep,inst.seed,σ,method,Q.LAMBDAS[j],secs,kernel_seconds,ng[j],Q.objective(p,F,S,y,Q.LAMBDAS[j],Bs[j])[1],val[j],st.iterations,haskey(st,:cg_iterations) ? st.cg_iterations : 0,status),','))
     end
    end
    mv(tmp, file; force=true)
    @printf("rep %d %s m=%d %s %.2fs converged %d/30\n",rep,p,m,method,secs,count(<(1e-4),ng));flush(stdout)
   end
  end
 end
end
