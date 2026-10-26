# Tables 2 and 3 (Section 5): kernel multinomial regression on Codon Usage
# (--data codon) or HAR (--data har) for m = ⌊n/64⌋, ..., ⌊n/4⌋, mean
# (standard deviation) over `reps` splits of time, test log-likelihood and
# test accuracy; also the 1-nearest-neighbor accuracy. Newton's method is run
# for m ≤ ⌊n/16⌋ (Codon) and m ≤ ⌊n/8⌋ (HAR).
#   julia --project -t 4 scripts/table_real_data.jl --data codon [--reps 20]
# Results are saved per split and m in results/realdata/, and finished ones
# are skipped.
# Output: tables/<data>.tex.
using QMMEReproducibility, Statistics
opt = options((data="codon", reps=20))
dir = resultdir("realdata")
const KS = (64, 32, 16, 8, 4)
newton_ks = opt.data == "har" ? (64, 32, 16, 8) : (64, 32, 16)
data = load_dataset(opt.data)
for seed in 1:opt.reps
    real_split(dir, opt.data, data, seed; ks=KS, newton_ks=newton_ks)
end

R = Dict((r.seed, r.k) => r for r in load_results(dir) if r.dataset == opt.data && r.seed <= opt.reps)
lines = [raw"\begin{tabular}{c l c c c}", raw"\hline",
         raw"\textbf{Projection Dimension} & \textbf{Method} & \textbf{Time (s)} & \textbf{Log-Likelihood} & \textbf{Accuracy (\%)}" * " \\\\",
         raw"\hline"]
for k in KS
    for (i, (meth, label)) in enumerate((("KMR_standard", "QMME (standard)"), ("KMR_full", "QMME (full)"), ("Newton", "Newton")))
        rows = [only(x for x in R[(s, k)].rows if x.method == meth) for s in 1:opt.reps if any(x -> x.method == meth, R[(s, k)].rows)]
        cells = isempty(rows) ? ["---", "---", "---"] :
            [meansd(getfield.(rows, :seconds)), meansd(getfield.(rows, :test_loglik)), meansd(100 .* getfield.(rows, :test_accuracy))]
        lead = i == 1 ? "\\multirow{3}{*}{\$\\lfloor n/$k \\rfloor\$}" : ""
        push!(lines, "$lead & $label & " * join(cells, " & ") * " \\\\")
    end
    push!(lines, raw"\hline")
end
push!(lines, raw"\end{tabular}")
knn = [R[(s, first(KS))].knn_accuracy for s in 1:opt.reps]
push!(lines, "% 1-NN test accuracy (%): " * meansd(100 .* knn))
savetable(opt.data, lines)
