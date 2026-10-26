# Baseline methods compared with QMME (Section 4): Newton's method, FISTA,
# adaptive gradient descent (AdaGD) and L-BFGS. All start from B = 0 and use
# the objectives in objectives.jl.
#
# Iteration convention (as in the QMME solvers): every iteration evaluates f
# and ∇f at the current point and stops when the gradient measure is below
# `tol` or after `max_iters` evaluations, so `max_iters` evaluations allow
# `max_iters - 1` updates. Trajectories record (objective, gradient norm, time)
# at every evaluation; times start at the first evaluation, so setup work
# (Lipschitz constants and so on) is excluded.

gradient_measure(g, measure) = measure === :l2 ? norm(g) : maximum(abs, g)
nvars(problem, y) = problem == "multinomial" ? size(y, 2) - 1 : 1
default_ridge(problem) = problem == "multinomial" ? 1e-6 : 1e-9

# Newton iterations at one λ from B (updated in place). The direction solves
# the dense Hessian system with `\`; the step is halved (at most 20 times)
# until the objective decreases, and the full step is taken if it never does.
function newton!(B, problem, F, S, y, λ; tol, measure=:l2, max_iters=100,
                 ridge=default_ridge(problem), trace=nothing)
    start = time_ns(); ng = Inf; iters = 0
    for iter = 1:max_iters
        f, g, w = objective(problem, F, S, y, λ, B)
        ng = gradient_measure(g, measure); iters = iter
        trace === nothing || push!(trace, (f, norm(g), (time_ns() - start) * 1e-9))
        (ng < tol || iter == max_iters) && break
        dir = reshape(hessian(problem, F, S, λ, w; ridge=ridge) \ vec(g), size(B))
        t = 1.0; accepted = false
        for _ = 0:20
            if objective(problem, F, S, y, λ, B .- t .* dir; gradient=false)[1] < f
                accepted = true; break
            end
            t /= 2
        end
        B .-= (accepted ? t : 1.0) .* dir
    end
    iters, ng
end

"""
    newton_path(problem, F, S, y, λs; tol, measure=:l2, max_iters=100, ridge)
        -> (Bs, stats)

Newton's method along the path λs with warm starts. `Bs[j]` is the solution at
λs[j]; `stats[j]` holds `iterations`, `gradient` and `converged`.
"""
function newton_path(problem, F, S, y, λs; tol, measure=:l2, max_iters=100, ridge=default_ridge(problem))
    B = zeros(size(F, 2), nvars(problem, y)); Bs = Matrix{Float64}[]; stats = NamedTuple[]
    for λ in λs
        iters, ng = newton!(B, problem, F, S, y, λ; tol=tol, measure=measure, max_iters=max_iters, ridge=ridge)
        push!(Bs, copy(B)); push!(stats, (lambda=λ, iterations=iters, gradient=ng, converged=ng < tol))
    end
    Bs, stats
end

"""Newton's method at one λ; returns (B, trace) with trace entries (f, ‖∇f‖₂, seconds)."""
function newton_trace(problem, F, S, y, λ; tol=1e-4, max_iters=1001, ridge=default_ridge(problem))
    B = zeros(size(F, 2), nvars(problem, y)); trace = Tuple{Float64,Float64,Float64}[]
    newton!(B, problem, F, S, y, λ; tol=tol, max_iters=max_iters, ridge=ridge, trace=trace)
    B, trace
end

# Lipschitz constant of ∇f: c‖F‖₂² + λ‖S‖₂, with c the bound on the second
# derivative of the loss (1/(√(2π)h), 1/4 and 1).
function lipschitz(problem, F, S, λ)
    c = problem == "quantile" ? 1 / (sqrt(2π) * H_BW) : problem == "logistic" ? 0.25 : 1.0
    c * eigmax(Symmetric(F' * F)) + λ * eigmax(Symmetric(S))
end

"""FISTA (Beck and Teboulle, 2009) with step 1/L; returns (B, trace)."""
function fista_trace(problem, F, S, y, λ; tol=1e-4, max_iters=1001)
    step = 1 / lipschitz(problem, F, S, λ)
    d = nvars(problem, y); γ = zeros(size(F, 2), d); δ = copy(γ); B = copy(γ); t = 1.0
    trace = Tuple{Float64,Float64,Float64}[]; start = time_ns()
    for iter = 1:max_iters
        tnext = (1 + sqrt(1 + 4t^2)) / 2
        @. B = γ + ((t - 1) / tnext) * (γ - δ); δ .= γ; t = tnext
        f, g, _ = objective(problem, F, S, y, λ, B)
        push!(trace, (f, norm(g), (time_ns() - start) * 1e-9))
        (norm(g) < tol || iter == max_iters) && break
        @. γ = B - step * g
    end
    B, trace
end

"""Adaptive gradient descent, Algorithm 1 of Malitsky and Mishchenko (2020),
initial step 1e-7; returns (B, trace)."""
function adagd_trace(problem, F, S, y, λ; tol=1e-4, max_iters=1001, t0=1e-7)
    d = nvars(problem, y); B = zeros(size(F, 2), d); Bprev = copy(B); gprev = copy(B)
    t = tprev = t0; θ = Inf
    trace = Tuple{Float64,Float64,Float64}[]; start = time_ns()
    for iter = 1:max_iters
        f, g, _ = objective(problem, F, S, y, λ, B)
        push!(trace, (f, norm(g), (time_ns() - start) * 1e-9))
        (norm(g) < tol || iter == max_iters) && break
        if iter > 1
            t = min(sqrt(1 + θ) * tprev, norm(B - Bprev) / (2 * norm(g - gprev)))
            θ = t / tprev; tprev = t
        end
        Bprev .= B; gprev .= g; B .-= t .* g
    end
    B, trace
end

"""L-BFGS from Optim.jl (memory 5, Hager–Zhang line search, no preconditioner),
stopped by the same gradient rule; returns (B, trace)."""
function lbfgs_trace(problem, F, S, y, λ; tol=1e-4, max_iters=1001)
    m = size(F, 2); d = nvars(problem, y)
    function fg!(f, g, x)
        val, grad, _ = objective(problem, F, S, y, λ, reshape(x, m, d); gradient=g !== nothing)
        g === nothing || copyto!(g, grad)
        f === nothing ? nothing : val
    end
    trace = Tuple{Float64,Float64,Float64}[]; start = Ref(UInt64(0))
    function callback(state)
        isempty(trace) && (start[] = time_ns())
        push!(trace, (state.f_x, norm(state.g_x), (time_ns() - start[]) * 1e-9))
        trace[end][2] < tol
    end
    opts = Optim.Options(iterations=max_iters - 1, g_abstol=0.0, f_abstol=0.0, f_reltol=0.0,
                         x_abstol=0.0, x_reltol=0.0, callback=callback)
    r = Optim.optimize(Optim.NLSolversBase.only_fg!(fg!), zeros(m * d), Optim.LBFGS(m=5), opts)
    reshape(Optim.minimizer(r), m, d), trace
end

"""Unpenalized linear model with intercept (the "Linear" reference lines of
Figure 4), fitted by Newton's method on the design [1 X]."""
function linear_fit(problem, X, y)
    D = hcat(ones(size(X, 1)), X)
    B = zeros(size(D, 2), nvars(problem, y))
    newton!(B, problem, D, zeros(size(D, 2), size(D, 2)), y, 0.0; tol=1e-6, max_iters=100, ridge=1e-9)
    B
end
