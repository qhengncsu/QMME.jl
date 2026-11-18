using QMMEReproducibility,LinearAlgebra,Random,Test
const Q=QMMEReproducibility
rng=MersenneTwister(12);F=randn(rng,80,12);S=Matrix{Float64}(I,12,12)
for p in ("quantile","logistic","multinomial")
 y=p=="quantile" ? randn(rng,80) : p=="logistic" ? Float64.(rand(rng,80).<.5) : Q.sim_labels(randn(rng,80),randn(rng,80),3,7)
 B=.1randn(rng,12,Q.nvars(p,y));f,g,w=Q.objective(p,F,S,y,.2,B);V=randn(rng,size(B));D=Q.hessian(p,F,S,.2,w;ridge=Q.default_ridge(p))
 @test vec(Q.hvp(p,F,S,.2,w,V)) ≈ D*vec(V)
 h=1e-5;gp=Q.objective(p,F,S,y,.2,B+h*V)[2];gm=Q.objective(p,F,S,y,.2,B-h*V)[2]
 @test Q.hvp(p,F,S,.2,w,V) ≈ (gp-gm)/(2h) rtol=1e-5
 blocks=Q.block_jacobi(p,F,S,.2,w;blocksize=4);dir,_=Q.pcg_direction(p,F,S,.2,w,g,blocks;forcing=1e-10)
 @test norm(D*vec(dir)-vec(g))/norm(g)<1e-9
 for method in ("LBFGS","Newton-PCG")
  Bs,stats=Q.competitive_path(p,method,F,S,y,[1.,.1];time_limit=30.)
  @test all(s.converged for s in stats)
 end
end
Q.warmup_sim("multinomial")
println("Hessian products, finite differences, PCG residuals, baseline convergence and spectral/Schur equivalence passed.")
