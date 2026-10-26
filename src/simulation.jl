# Simulated data of Section 4 and the solution-path experiments of Sections
# 4.2 (Figure 4) and 4.3 (Figure 5).
#
# X has n = 2^14 rows of N(0, Σ), Σ_ij = 0.5^|i-j|, p = 50; two nonlinear
# signals a(X), b(X) drive the quantile, logistic and multinomial responses.
# Each replication draws a training set and an independent validation set of
# size n/4. Landmarks are the first m entries of one random permutation, so the
# landmark sets are nested in m. The bandwidth σ is the median pairwise
# distance of the training X (median heuristic).

const N_SIM = 2^14
const LAMBDAS = 10.0 .^ range(1, -5, length=30)   # path from 10 to 1e-5

function signals(X)
    tail = vec(sum(abs2, @view(X[:, 6:end]); dims=2))
    a = @. -4 + sin(X[:, 1]) + X[:, 2] * X[:, 3] + X[:, 4]^3 - abs(X[:, 5]) + 0.1 * tail
    b = @. 4 + cos(X[:, 1]) + X[:, 2] * X[:, 4] + X[:, 5]^3 - abs(X[:, 3]) - 0.1 * tail
    a, b
end

"""Instance of replication `rep` for experiment `group` (1 = solution path, 2 = increasing q)."""
function sim_instance(rep, group; n=N_SIM)
    seed = 202610200 + 1000group + rep; rng = MersenneTwister(seed); p = 50
    L = cholesky(Symmetric([0.5^abs(i - j) for i = 1:p, j = 1:p])).L
    X = randn(rng, n, p) * L'; Xv = randn(rng, n ÷ 4, p) * L'
    a, b = signals(X); av, bv = signals(Xv)
    yq = a + rand(rng, TDist(1.5), n)
    yl = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-a)))
    ylv = Float64.(rand(rng, n ÷ 4) .< 1 ./ (1 .+ exp.(-av)))
    (; X, Xv, a, b, av, bv, yq, yl, ylv, idx=randperm(rng, n), seed)
end

"""One-hot labels with q classes: logits (a, b, 0, ..., 0)."""
function sim_labels(a, b, q, seed)
    rng = MersenneTwister(seed); Y = zeros(length(a), q)
    for i in eachindex(a)
        z = zeros(q); z[1] = a[i]; z[2] = b[i]; z .-= maximum(z); p = exp.(z); p ./= sum(p)
        Y[i, rand(rng, Categorical(p))] = 1.0
    end
    Y
end

"""Gaussian kernel matrix exp(-‖x_i - z_j‖²/(2σ²))."""
function rbf(X, Z, σ)
    D = sum(abs2, X; dims=2) .+ sum(abs2, Z; dims=2)' .- 2 .* (X * Z')
    @. D = exp(-max(D, 0.0) / (2σ^2))
end

"""Median of the pairwise Euclidean distances between the rows of X."""
function median_distance(X)
    sq = vec(sum(abs2, X; dims=2)); D2 = sq .+ sq' .- 2 .* (X * X')
    median!([sqrt(max(D2[i, j], 0.0)) for j in axes(D2, 2) for i in 1:j-1])
end

const MULTINOMIAL_METHODS = ["Alg2-standard", "Alg3-standard", "Newton", "Alg2-full", "Alg3-full"]

# Solution path of one method (‖∇f‖₂ < 1e-4, at most 1000 iterations per λ).
function fit_path(problem, method, F, S, y)
    kw = (tol=1e-4, max_iters=1000)
    if method == "Newton"
        return newton_path(problem, F, S, y, LAMBDAS; kw...)
    end
    qkw = (kw..., grad_measure=:l2)
    if problem == "quantile"
        Bs, _, stats = KernelQuantileQMME.kernel_quantile(y, F, S, TAU, LAMBDAS; h=H_BW, qkw...)
        return [permutedims(Bs[j:j, :]) for j in axes(Bs, 1)], stats
    elseif problem == "logistic"
        Bs, _, stats = KernelLogisticQMME.kernel_logistic(y, F, S, LAMBDAS; qkw...)
        return [permutedims(Bs[j:j, :]) for j in axes(Bs, 1)], stats
    end
    mod = startswith(method, "Alg2") ? KernelMultinomialAlg2 : KernelMultinomialAlg3
    f = endswith(method, "full") ? mod.KernelMultinomial_full : mod.KernelMultinomial
    Bs, _, stats = f(y, F, S, LAMBDAS; qkw...)
    [Bs[j, :, :] for j in axes(Bs, 1)], stats
end

# Compiles every method on a tiny problem before anything is timed, and checks
# that Algorithms 2 and 3 give the same multinomial path.
const WARMED = Set{String}()
function warmup_sim(problem)
    problem in WARMED && return
    rng = MersenneTwister(4); F = randn(rng, 40, 6); S = Matrix{Float64}(I, 6, 6)
    y = problem == "quantile" ? randn(rng, 40) : problem == "logistic" ? Float64.(rand(rng, 40) .< 0.5) :
        sim_labels(randn(rng, 40), randn(rng, 40), 3, 19)
    methods = problem == "multinomial" ? MULTINOMIAL_METHODS : ["QMME", "Newton"]
    B = Dict(meth => fit_path(problem, meth, F, S, y)[1] for meth in methods)
    if problem == "multinomial"
        for p in ("standard", "full")
            @assert all(isapprox.(B["Alg2-"*p], B["Alg3-"*p]; rtol=1e-5, atol=2e-5)) "Algorithms 2 and 3 disagree"
        end
    end
    push!(WARMED, problem); nothing
end

"""
    sim_setting(dir, group, rep, problem, σ, m, q, inst, y, yv; methods)

Fits the path of every method in `methods` (skipping those already saved in
`dir`) for one replication and one m (or q), and saves one record per method:
time (kernel construction + path), best validation metric over λ, and the
number of λ whose solution has ‖∇f‖₂ < 1e-4, recomputed independently.
"""
function sim_setting(dir, group, rep, problem, σ, m, q, inst, y, yv; methods)
    file(meth) = joinpath(dir, "$(group)_$(problem)_m$(m)_q$(q)_$(meth)_rep$(rep).jls")
    todo = filter(meth -> !isfile(file(meth)), methods)
    isempty(todo) && return
    warmup_sim(problem)
    idx = inst.idx[1:m]
    t = time_ns(); F = rbf(inst.X, inst.X[idx, :], σ); S = Matrix(Symmetric(F[idx, :])); kernel_s = (time_ns() - t) * 1e-9
    Fv = rbf(inst.Xv, inst.X[idx, :], σ)
    # rotate the order across replications so no method always runs first
    for meth in circshift(todo, rep - 1)
        GC.gc(); t = time_ns(); Bs, stats = fit_path(problem, meth, F, S, y); solver_s = (time_ns() - t) * 1e-9
        full = endswith(meth, "full")
        gn = [norm(objective(problem, F, S, y, LAMBDAS[j], Bs[j]; full=full)[2]) for j in eachindex(LAMBDAS)]
        val = [validation(problem, Bs[j], Fv, yv; full=full) for j in eachindex(LAMBDAS)]
        r = (group=group, rep=rep, problem=problem, m=m, q=q, method=meth, sigma=σ,
             seconds=solver_s + kernel_s, best_validation=problem == "quantile" ? minimum(val) : maximum(val),
             converged=count(<(1e-4), gn), iterations=sum(s.iterations for s in stats))
        save_result(file(meth), r)
        @printf("%-8s rep %2d %-11s m=%-5d q=%-2d %-14s %8.2f s  converged %d/%d\n",
                group, rep, problem, m, q, meth, r.seconds, r.converged, length(LAMBDAS)); flush(stdout)
    end
end

"""Solution-path experiment (Figure 4) for replication `rep` and one problem."""
function sim_path(dir, rep, problem; newton=true, ms=problem == "multinomial" ? 2 .^ (2:11) : 2 .^ (3:12))
    inst = sim_instance(rep, 1); σ = cached(() -> median_distance(inst.X), joinpath(mkpath(joinpath(dir, "sigma")), "path_rep$(rep).jls"))
    y  = problem == "quantile" ? inst.yq : problem == "logistic" ? inst.yl : sim_labels(inst.a, inst.b, 3, inst.seed + 30)
    yv = problem == "quantile" ? inst.av : problem == "logistic" ? inst.ylv : sim_labels(inst.av, inst.bv, 3, inst.seed + 31)
    lin = joinpath(dir, "linear_$(problem)_rep$(rep).jls")
    isfile(lin) || save_result(lin, (problem=problem, rep=rep,
        validation=validation(problem, linear_fit(problem, inst.X, y), hcat(ones(size(inst.Xv, 1)), inst.Xv), yv)))
    methods = problem == "multinomial" ? MULTINOMIAL_METHODS : ["QMME", "Newton"]
    newton || (methods = filter(!=("Newton"), methods))
    for m in ms
        sim_setting(dir, "path", rep, problem, σ, m, 3, inst, y, yv; methods=methods)
    end
end

"""Increasing-q experiment (Figure 5) for replication `rep`, m = 2^9."""
function sim_classes(dir, rep; newton=true, qs=3:10, m=2^9)
    inst = sim_instance(rep, 2); σ = cached(() -> median_distance(inst.X), joinpath(mkpath(joinpath(dir, "sigma")), "classes_rep$(rep).jls"))
    methods = newton ? MULTINOMIAL_METHODS : filter(!=("Newton"), MULTINOMIAL_METHODS)
    for q in qs
        y = sim_labels(inst.a, inst.b, q, inst.seed + 10q); yv = sim_labels(inst.av, inst.bv, q, inst.seed + 10q + 1)
        sim_setting(dir, "classes", rep, "multinomial", σ, m, q, inst, y, yv; methods=methods)
    end
end

"""
    trajectory_instance(n, m; seed=20261020, σ=20) -> (F, S, ys)

The single instance of Section 4.1 (Figures 1-3 and the Newton-CG table):
the same design and signals, σ by the median heuristic (median pairwise
distance of the rows of X; `σ` overrides it), m random landmarks; `ys` maps
each problem to its response.
"""
function trajectory_instance(n, m; seed=20261020, σ=nothing)
    rng = MersenneTwister(seed); p = 50
    X = randn(rng, n, p) * cholesky(Symmetric([0.5^abs(i - j) for i = 1:p, j = 1:p])).U
    a, b = signals(X)
    yq = a + rand(rng, TDist(1.5), n)
    yl = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-a)))
    Y = zeros(n, 3)
    for i = 1:n
        c = max(0.0, a[i], b[i]); p1 = exp(a[i] - c); p2 = exp(b[i] - c); p3 = exp(-c)
        u = rand(rng) * (p1 + p2 + p3); Y[i, u < p1 ? 1 : (u < p1 + p2 ? 2 : 3)] = 1
    end
    idx = randperm(rng, n)[1:m]
    σ === nothing && (σ = median_distance(X))
    F = rbf(X, X[idx, :], σ)
    F, Matrix(Symmetric(F[idx, :])), Dict("quantile" => yq, "logistic" => yl, "multinomial" => Y)
end
