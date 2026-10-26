# Figures 1 and 2 (Section 4.1): objective and gradient-norm trajectories of
# QMME, FISTA, AdaGD, L-BFGS and Newton's method on one instance
# (n = 2^14, m = 2^11, σ by the median heuristic, λ = 1e-4). Every method starts from zero and
# stops at ‖∇f‖₂ < 1e-4 or after 1000 updates.
#   julia --project -t 4 scripts/fig1_fig2_trajectories.jl [--n 16384] [--m 2048]
# Output: figures/ObjTraj.pdf, figures/GradNormTraj.pdf (and .png); the
# trajectories are saved in results/trajectories/.
using QMMEReproducibility, Plots, Measures
ENV["GKSwstype"] = "100"
opt = options((n=2^14, m=2^11))
const λ = 1e-4
const PROBLEMS = ("quantile", "logistic", "multinomial")
const TITLES = Dict("quantile" => "Quantile Regression", "logistic" => "Logistic Regression", "multinomial" => "Multinomial Regression")
# (key, label, color, line style) of every curve, in legend order
methods(p) = vcat(p == "multinomial" ? [("QMME2", "QMME (Alg. 2)", "red", :dash), ("QMME3", "QMME (Alg. 3)", "red", :solid)] :
                                       [("QMME", "QMME", "red", :solid)],
                  [("FISTA", "FISTA", "blue", :solid), ("AdaGD", "Adaptive GD", "green", :solid),
                   ("LBFGS", "L-BFGS", "#00A6B2", :solid), ("Newton", "Newton", "purple", :solid)])

# Trajectory of one method as (objective, gradient norm, seconds) vectors; at
# most 1001 evaluations = 1000 updates.
function trajectory(p, key, F, S, y; max_iters=1001)
    kw = (tol=1e-4, max_iters=max_iters)
    if startswith(key, "QMME")
        r = p == "quantile" ? KernelQuantileQMME.kernel_quantile(y, F, S, TAU, λ; h=H_BW, grad_measure=:l2, kw...) :
            p == "logistic" ? KernelLogisticQMME.kernel_logistic(y, F, S, λ; grad_measure=:l2, kw...) :
            (key == "QMME2" ? KernelMultinomialAlg2 : KernelMultinomialAlg3).KernelMultinomial(y, F, S, λ; grad_measure=:l2, kw...)
        obj, gn, t = r[2], r[3], r[4]
    else
        f = Dict("FISTA" => fista_trace, "AdaGD" => adagd_trace, "LBFGS" => lbfgs_trace, "Newton" => newton_trace)[key]
        tr = f(p, F, S, y, λ; kw...)[2]
        obj, gn, t = first.(tr), getindex.(tr, 2), last.(tr)
    end
    (objective=obj, gradient_norm=gn, seconds=t .- t[1])    # time 0 at the first evaluation
end

# compile every method on a small instance first
let (F, S, ys) = trajectory_instance(400, 20)
    for p in PROBLEMS, (key, _) in methods(p); trajectory(p, key, F, S, ys[p]; max_iters=5); end
end
dir = resultdir("trajectories")
F, S, ys = trajectory_instance(opt.n, opt.m)
T = Dict((p, key) => cached(joinpath(dir, "$(p)_$(key).jls")) do
            GC.gc(); r = trajectory(p, key, F, S, ys[p])
            println(rpad(p, 12), rpad(key, 7), length(r.objective) - 1, " updates, ", round(r.seconds[end]; digits=2), " s"); r
        end for p in PROBLEMS for (key, _) in methods(p))

# Each time axis ends at the shortest runtime among the methods that made 1000 updates.
timelimit(p) = minimum(T[(p, k)].seconds[1001] for (k, _) in methods(p) if length(T[(p, k)].seconds) > 1000)

function draw(field, ylabel; inset)
    plot(layout=(2, 3), size=(1200, 800), margin=5mm,
         inset_subplots=inset ? [(i, bbox(0.43, 0.69, 0.48, 0.21)) for i in 1:3] : nothing)
    for (i, p) in enumerate(PROBLEMS), (key, label, color, style) in methods(p)
        r = T[(p, key)]; v = log.(getfield(r, field))
        plot!(0:length(v)-1, v; subplot=i, label=label, color=color, linestyle=style, title=TITLES[p],
              xlabel="Iteration", ylabel=i == 1 ? ylabel : "", legend=i == 2 ? :none : :right, xlims=(0, 1000))
        plot!(r.seconds, v; subplot=i + 3, label=label, color=color, linestyle=style, legend=:none,
              xlabel="Time (seconds)", ylabel=i == 1 ? ylabel : "", xlims=(0, timelimit(p)))
    end
    inset || return
    for (i, p) in enumerate(PROBLEMS)            # the first 20 iterations
        early = [log.(T[(p, k)].objective[1:min(21, end)]) for (k, _) in methods(p)]
        lo, hi = extrema(reduce(vcat, early))
        for ((key, _, color, style), v) in zip(methods(p), early)
            plot!(0:length(v)-1, v; subplot=6 + i, color=color, linestyle=style, linewidth=1, label="", legend=false,
                  xlims=(0, 20), xticks=[0, 10, 20], xlabel="", ylabel="", title="", tickfontsize=6,
                  yticks=round.([lo, (lo + hi) / 2, hi]; digits=2), ylims=(lo - 0.04(hi - lo), hi + 0.04(hi - lo)),
                  framestyle=:box, background_color_inside=:white, foreground_color_border=:gray, gridalpha=0.12, widen=true)
        end
        annotate!(10, (lo + hi) / 2, text("First 20 iterations", 6, :black); subplot=6 + i)
    end
end
draw(:objective, "Objective (log scale)"; inset=true);              savefigure("ObjTraj")
draw(:gradient_norm, "Gradient Norm (log scale)"; inset=false);     savefigure("GradNormTraj")
