# Figure 4 (Section 4.2): total time of the solution path (30 λ from 10 to
# 1e-5, warm starts, ‖∇f‖₂ < 1e-4) and best validation accuracy, as the
# projection dimension m grows; QMME versus Newton's method, σ by the median
# heuristic. Newton's method is run in the first `newton-reps` replications.
#   julia --project -t 4 scripts/fig4_solution_path.jl [--reps 20] [--newton-reps 10]
# Results are saved per replication and setting in results/simulation/, and
# finished settings are skipped.
# Output: figures/path.pdf (and .png).
using QMMEReproducibility, Plots, Measures, Statistics
ENV["GKSwstype"] = "100"
opt = options((reps=20, newton_reps=10))
dir = resultdir("simulation")
for rep in 1:opt.reps, p in ("quantile", "logistic", "multinomial")
    sim_path(dir, rep, p; newton=rep <= opt.newton_reps)
end

R = filter(r -> haskey(r, :group) && r.group == "path", load_results(dir))
# replications × m matrix of `field` for one problem and method
function grid(p, meth, ms, field)
    reps = meth == "Newton" ? (1:opt.newton_reps) : (1:opt.reps)
    [only(getfield(r, field) for r in R if r.problem == p && r.method == meth && r.m == m && r.rep == k) for k in reps, m in ms]
end
linear(p) = mean(r.validation for r in load_results(dir) if haskey(r, :validation) && r.problem == p && r.rep <= opt.reps)
band!(x, A; kw...) = plot!(x, vec(mean(A; dims=1)); yerr=vec(std(A; dims=1)) ./ sqrt(size(A, 1)), kw...)
@assert all(r.converged == length(LAMBDAS) for r in R) "a path did not converge"

ms = 2 .^ (3:12); mms = 2 .^ (2:11)
MULTI = [("Alg2-standard", "QMME (Alg. 2)", "red", :dash), ("Alg3-standard", "QMME (Alg. 3)", "red", :solid),
         ("Newton", "Newton", "blue", :solid), ("Alg2-full", "QMME (Alg. 2, full)", "green", :dash),
         ("Alg3-full", "QMME (Alg. 3, full)", "green", :solid)]
plot(layout=(2, 3), size=(1200, 800), margin=5mm, legendfontsize=9)
for (i, p, title) in ((1, "quantile", "Quantile Regression"), (2, "logistic", "Logistic Regression"))
    band!(3:12, grid(p, "QMME", ms, :seconds); label="QMME", color="red", markerstrokecolor=:red,
          subplot=i, title=title, xlabel="log2(m)", ylabel="Time (seconds)")
    band!(3:12, grid(p, "Newton", ms, :seconds); label="Newton", color="blue", markerstrokecolor=:blue,
          subplot=i, legend=i == 1 ? :topleft : :none)
    band!(3:12, grid(p, "QMME", ms, :best_validation); label="Kernel", color="red", markerstrokecolor=:red,
          subplot=i + 3, xlabel="log2(m)", ylabel=p == "quantile" ? "MAD" : "Log-likelihood")
    hline!([linear(p)]; label="Linear", color="blue", linestyle=:dash, subplot=i + 3, legend=i == 1 ? :best : :none)
end
for (meth, label, color, style) in MULTI
    band!(2:11, grid("multinomial", meth, mms, :seconds); label=label, color=color, linestyle=style, markerstrokecolor=Symbol(color),
          subplot=3, title="Multinomial Regression", xlabel="log2(m)", ylabel="Time (seconds)")
    meth == "Newton" && continue
    band!(2:11, grid("multinomial", meth, mms, :best_validation); label=label, color=color, linestyle=style,
          markerstrokecolor=Symbol(color), subplot=6, xlabel="log2(m)", ylabel="Log-likelihood")
end
hline!([linear("multinomial")]; label="Linear", color="blue", linestyle=:dash, subplot=6)
plot!(subplot=3, legend=:topleft); plot!(subplot=6, legend=:bottomright)
savefigure("path")
