# Figure 6 (Section 5): validation and test log-likelihood along the tuning
# path of one split, Codon Usage (m = ⌊n/8⌋ = 1302) and HAR (m = 1029),
# standard parameterization. The path is computed with max|∇f| < 1e-4 (at
# most 10000 iterations per λ) so that small-λ solutions are not left at the
# warm start.
#   julia --project -t 4 scripts/fig6_real_path.jl [--seed 10]
# Output: figures/realpath.pdf (and .png); curves in results/realpath/.
using QMMEReproducibility, Plots, Measures
ENV["GKSwstype"] = "100"
opt = options((seed=10,))
dir = resultdir("realpath")
plot(layout=(1, 2), size=(1200, 600), margin=5mm, legendfontsize=9)
for (i, (name, title)) in enumerate((("codon", "Codon Usage"), ("har", "HAR")))
    c = cached(() -> real_path_curve(name, load_dataset(name), opt.seed, 8), joinpath(dir, "$(name)_seed$(opt.seed).jls"))
    x = log10.(c.lambda)
    plot!(x, c.val; subplot=i, label="Validation Log-likelihood", color="blue", marker=:square, markersize=4,
          markerstrokecolor=:blue, title="$title (m = $(c.m))", xlabel="log10(lambda)", ylabel="Log-likelihood", xflip=true)
    plot!(x, c.test; subplot=i, label="Test Log-likelihood", color="red", marker=:square, markersize=4,
          markerstrokecolor=:red, legend=:bottomright)
end
savefigure("realpath")
