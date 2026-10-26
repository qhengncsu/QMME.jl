# Figure 3 (Section 4.1): QMME with the extrapolation coefficient
# β_k = l/(l+2) of Algorithm 1 versus β_k = 0, on the instance of Figure 1.
# Both start from zero and stop at ‖∇f‖₂ < 1e-4 or after 1000 updates.
#   julia --project -t 4 scripts/fig3_momentum.jl [--n 16384] [--m 2048]
# Output: figures/momentum.pdf (and .png); trajectories in results/momentum/.
using QMMEReproducibility, Plots, Measures
ENV["GKSwstype"] = "100"
opt = options((n=2^14, m=2^11))
const λ = 1e-4
const RUNS = [("quantile", "QMME", "", :solid), ("logistic", "QMME", "", :solid),
              ("multinomial", "Alg2", " (Alg. 2)", :dash), ("multinomial", "Alg3", " (Alg. 3)", :solid)]

function qmme(p, alg, F, S, y, momentum; max_iters=1001)
    kw = (tol=1e-4, max_iters=max_iters, grad_measure=:l2, momentum=momentum)
    r = p == "quantile" ? KernelQuantileQMME.kernel_quantile(y, F, S, TAU, λ; h=H_BW, kw...) :
        p == "logistic" ? KernelLogisticQMME.kernel_logistic(y, F, S, λ; kw...) :
        (alg == "Alg2" ? KernelMultinomialAlg2 : KernelMultinomialAlg3).KernelMultinomial(y, F, S, λ; kw...)
    (objective=r[2], gradient_norm=r[3], seconds=r[4] .- r[4][1])
end

let (F, S, ys) = trajectory_instance(400, 20)      # compile first
    for (p, alg) in RUNS, mo in (:adaptive, :none); qmme(p, alg, F, S, ys[p], mo; max_iters=5); end
end
dir = resultdir("momentum")
F, S, ys = trajectory_instance(opt.n, opt.m)
T = Dict((p, alg, mo) => cached(joinpath(dir, "$(p)_$(alg)_$(mo).jls")) do
            GC.gc(); r = qmme(p, alg, F, S, ys[p], mo)
            println(rpad(p, 12), rpad(alg, 5), rpad(mo, 9), length(r.objective) - 1, " updates, ", round(r.seconds[end]; digits=2), " s"); r
        end for (p, alg) in RUNS for mo in (:adaptive, :none))

titles = ["Quantile Regression", "Logistic Regression", "Multinomial Regression"]
plot(layout=(2, 3), size=(1200, 800), margin=5mm, legendfontsize=9)
for (p, alg, lab, style) in RUNS
    i = findfirst(==(p), ("quantile", "logistic", "multinomial"))
    for (row, x, xl) in ((0, :iteration, "Iteration"), (3, :seconds, "Time (seconds)"))
        for (mo, color, suffix) in ((:adaptive, "red", ""), (:none, "black", ", β = 0"))
            r = T[(p, alg, mo)]
            xs = x == :iteration ? (0:length(r.objective)-1) : r.seconds
            plot!(xs, log.(r.gradient_norm); subplot=i + row, label="QMME" * lab * suffix, color=color, linestyle=style,
                  title=row == 0 ? titles[i] : "", xlabel=xl, ylabel=i == 1 ? "Gradient Norm (log scale)" : "",
                  legend=row == 0 ? :topright : :none)
        end
    end
end
for (i, p) in enumerate(("quantile", "logistic", "multinomial"))   # time axes start at 0
    plot!(subplot=i, xlims=(0, maximum(length(r.objective) - 1 for ((q, _, _), r) in T if q == p)))
    plot!(subplot=i + 3, xlims=(0, 1.02 * maximum(r.seconds[end] for ((q, _, _), r) in T if q == p)))
end
savefigure("momentum")
