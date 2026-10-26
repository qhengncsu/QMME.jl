# Figure 5 (Section 4.3): kernel multinomial regression with q = 3, ..., 10
# classes at m = 2^9: total time of the solution path and best validation
# log-likelihood; QMME (Algorithms 2 and 3, standard and full) versus Newton's
# method, which is run in the first `newton-reps` replications.
#   julia --project -t 4 scripts/fig5_increase_q.jl [--reps 20] [--newton-reps 10]
# Results are saved per replication and setting in results/simulation/, and
# finished settings are skipped.
# Output: figures/increaseq.pdf (and .png).
using QMMEReproducibility, Plots, Measures, Statistics
ENV["GKSwstype"] = "100"
opt = options((reps=20, newton_reps=10))
dir = resultdir("simulation")
for rep in 1:opt.reps
    sim_classes(dir, rep; newton=rep <= opt.newton_reps)
end

R = filter(r -> haskey(r, :group) && r.group == "classes", load_results(dir))
function grid(meth, qs, field)
    reps = meth == "Newton" ? (1:opt.newton_reps) : (1:opt.reps)
    [only(getfield(r, field) for r in R if r.method == meth && r.q == q && r.rep == k) for k in reps, q in qs]
end
band!(x, A; kw...) = plot!(x, vec(mean(A; dims=1)); yerr=vec(std(A; dims=1)) ./ sqrt(size(A, 1)), kw...)
@assert all(r.converged == length(LAMBDAS) for r in R) "a path did not converge"

qs = 3:10
plot(layout=(1, 2), size=(1200, 600), margin=5mm, legendfontsize=9)
for (meth, label, color, style) in [("Alg2-standard", "QMME (Alg. 2)", "red", :dash), ("Alg3-standard", "QMME (Alg. 3)", "red", :solid),
                                    ("Newton", "Newton", "blue", :solid), ("Alg2-full", "QMME (Alg. 2, full)", "green", :dash),
                                    ("Alg3-full", "QMME (Alg. 3, full)", "green", :solid)]
    kw = (label=label, color=color, linestyle=style, markerstrokecolor=Symbol(color), xlabel="q")
    band!(qs, grid(meth, qs, :seconds); subplot=1, ylabel="Time (seconds)", kw...)
    meth == "Newton" || band!(qs, grid(meth, qs, :best_validation); subplot=2, ylabel="Log-likelihood", kw...)
end
plot!(subplot=1, legend=:topleft); plot!(subplot=2, legend=:topright)
savefigure("increaseq")
