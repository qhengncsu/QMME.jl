# Quick checks (about a minute): gradients and Hessians against finite
# differences, and QMME against Newton's method on small problems.
#   julia --project -t 4 test/runtests.jl
using QMMEReproducibility, Test, LinearAlgebra, Random

F, S, ys = trajectory_instance(400, 30)

@testset "gradient and Hessian, $p" for p in ("quantile", "logistic", "multinomial")
    rng = MersenneTwister(1); d = p == "multinomial" ? 2 : 1; λ = 1e-2
    B = 0.1 * randn(rng, 30, d); V = randn(rng, 30, d); h = 1e-6
    f, g, w = objective(p, F, S, ys[p], λ, B)
    fd = (objective(p, F, S, ys[p], λ, B + h * V)[1] - objective(p, F, S, ys[p], λ, B - h * V)[1]) / 2h
    @test isapprox(fd, dot(g, V); rtol=1e-6)
    Hv = (objective(p, F, S, ys[p], λ, B + h * V)[2] - objective(p, F, S, ys[p], λ, B - h * V)[2]) / 2h
    @test isapprox(hessian(p, F, S, λ, w; ridge=0.0) * vec(V), vec(Hv); rtol=1e-5)
    if p == "multinomial"     # full parameterization (q columns)
        Bf = 0.1 * randn(rng, 30, 3); Vf = randn(rng, 30, 3)
        _, gf, _ = objective(p, F, S, ys[p], λ, Bf; full=true)
        fdf = (objective(p, F, S, ys[p], λ, Bf + h * Vf; full=true)[1] - objective(p, F, S, ys[p], λ, Bf - h * Vf; full=true)[1]) / 2h
        @test isapprox(fdf, dot(gf, Vf); rtol=1e-6)
    end
end

@testset "QMME and Newton reach the same path, $p" for p in ("quantile", "logistic", "multinomial")
    λs = [1.0, 1e-1, 1e-2]
    Bn, _ = newton_path(p, F, S, ys[p], λs; tol=1e-8)
    if p == "quantile"
        Bq = KernelQuantileQMME.kernel_quantile(ys[p], F, S, TAU, λs; h=H_BW, tol=1e-8, max_iters=20000, grad_measure=:l2)[1]
        Bq = [permutedims(Bq[j:j, :]) for j in axes(Bq, 1)]
    elseif p == "logistic"
        Bq = KernelLogisticQMME.kernel_logistic(ys[p], F, S, λs; tol=1e-8, max_iters=20000, grad_measure=:l2)[1]
        Bq = [permutedims(Bq[j:j, :]) for j in axes(Bq, 1)]
    else
        r2 = KernelMultinomialAlg2.KernelMultinomial(ys[p], F, S, λs; tol=1e-8, max_iters=20000, grad_measure=:l2)[1]
        r3 = KernelMultinomialAlg3.KernelMultinomial(ys[p], F, S, λs; tol=1e-8, max_iters=20000, grad_measure=:l2)[1]
        @test isapprox(r2, r3; rtol=1e-5)
        Bq = [r3[j, :, :] for j in axes(r3, 1)]
    end
    for j in eachindex(λs)
        fq = objective(p, F, S, ys[p], λs[j], Bq[j])[1]; fn = objective(p, F, S, ys[p], λs[j], Bn[j])[1]
        @test isapprox(fq, fn; rtol=1e-9)
    end
end

@testset "first-order baselines decrease the objective, $p" for p in ("quantile", "logistic", "multinomial")
    for tr in (fista_trace, adagd_trace, lbfgs_trace)
        _, t = tr(p, F, S, ys[p], 1e-2; max_iters=50)
        @test t[end][1] < t[1][1]
    end
end
