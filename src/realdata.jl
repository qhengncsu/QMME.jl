# Real data (Section 5): Codon Usage and HAR from the UCI Machine Learning
# Repository, kernel multinomial regression by QMME (Algorithm 3, standard and
# full parameterization), Newton's method and the 1-nearest-neighbor classifier.
#
# Protocol for one split (seed) and one projection dimension m:
#   * 60/20/20 train/validation/test split of the instances (Codon) or of the
#     30 subjects (HAR); features standardized with train+validation statistics;
#   * σ = median pairwise distance of the standardized train+validation rows
#     (a random 20000 of them when there are more);
#   * m class-stratified landmarks among the training rows (tuning) and among
#     the train+validation rows (final model);
#   * tuning: 30 λ from 10 to 1e-5 with warm starts, max|∇f| < 1e-2, at most
#     500 iterations per λ, λ chosen by the validation log-likelihood;
#   * final model: refit on train+validation at the chosen λ (at most 1000
#     iterations), test log-likelihood and accuracy.
# Reported times include every step except the 1-NN baseline.

const DATADIR = joinpath(ROOT, "data")
const UCI = Dict("codon" => "https://archive.ics.uci.edu/static/public/577/codon+usage.zip",
                 "har"   => "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip")

function download_raw(name)
    file = joinpath(mkpath(joinpath(DATADIR, "raw")), name * ".zip")
    isfile(file) || (println("Downloading ", UCI[name]); Downloads.download(UCI[name], file))
    file
end
zipentries(io) = (z = ZipFile.Reader(io); Dict(f.name => read(f) for f in z.files))

"""
    load_dataset(name) -> (X, labels, groups)

"codon": 13,026 organisms (the two rows with malformed codon frequencies are
dropped), 64 codon frequencies, 11 kingdoms; groups = nothing.
"har": 10,299 windows (original training and test files merged), 561
features, 6 activities; groups = subject (1-30). Feature values are rounded
to 7 significant digits, the precision of the data file used for the paper.
"""
function load_dataset(name)
    files = zipentries(IOBuffer(read(download_raw(name))))
    if name == "codon"
        t = CSV.File(IOBuffer(files["codon_usage.csv"]); types=String)
        cols = propertynames(t)
        num(v) = v === missing ? NaN : something(tryparse(Float64, v), NaN)
        X = [num(getproperty(r, cols[j])) for r in t, j = 6:69]
        keep = vec(all(isfinite, X; dims=2))
        return X[keep, :], String.([r.Kingdom for r in t][keep]), nothing
    elseif name == "har"
        inner = zipentries(IOBuffer(files["UCI HAR Dataset.zip"]))
        txt(f) = String(inner["UCI HAR Dataset/" * f])
        nums(f) = [parse.(Float64, split(l)) for l in eachline(IOBuffer(txt(f)))]
        X = reduce(vcat, permutedims.(vcat(nums("train/X_train.txt"), nums("test/X_test.txt"))))
        X = parse.(Float64, [@sprintf("%.7g", v) for v in X])
        act = Dict(parse(Int, a) => b for (a, b) in split.(eachline(IOBuffer(txt("activity_labels.txt")))))
        y = vcat(parse.(Int, eachline(IOBuffer(txt("train/y_train.txt")))), parse.(Int, eachline(IOBuffer(txt("test/y_test.txt")))))
        g = vcat(parse.(Int, eachline(IOBuffer(txt("train/subject_train.txt")))), parse.(Int, eachline(IOBuffer(txt("test/subject_test.txt")))))
        return X, [String(act[v]) for v in y], g
    end
    error("unknown dataset $name")
end

"""Projection dimension ⌊n/k⌋, with n = 80% of the instances (train + validation)."""
projection_dim(N, k) = (N * 6 ÷ 10 + N * 2 ÷ 10) ÷ k

# m class-stratified landmarks among the rows of the one-hot Y: each class gets
# ⌊m·share⌋ rows, the remainder goes to the largest fractional parts; sampled
# without replacement, then shuffled. Returns row indices of Y.
function stratified_landmarks(Y, m, rng)
    y = [findfirst(==(1), r) for r in eachrow(Y)]
    classes = unique(y); share = [count(==(c), y) for c in classes] ./ length(y)
    k = floor.(Int, share .* m)
    rem = m - sum(k)
    rem > 0 && (k[partialsortperm(share .* m .- k, 1:rem; rev=true)] .+= 1)
    idx = Int[]
    for (i, c) in enumerate(classes)
        pool = findall(==(c), y)
        append!(idx, sample(rng, pool, min(k[i], length(pool)); replace=false))
    end
    shuffle!(rng, idx)
end

"""
    prepare_split(X, labels, groups, seed, m)

Split, standardization, bandwidth, landmarks and kernel matrices for one
seed and one m; also returns the time of every step.
"""
function prepare_split(X, labels, groups, seed, m)
    classes = sort(unique(labels)); lab = [findfirst(==(v), classes) for v in labels]
    Y = zeros(length(lab), length(classes)); for i in eachindex(lab); Y[i, lab[i]] = 1.0; end
    N = size(X, 1)
    if groups === nothing
        perm = randperm(MersenneTwister(seed), N); nt = floor(Int, 0.6N); nv = floor(Int, 0.2N)
        train, val, test = perm[1:nt], perm[nt+1:nt+nv], perm[nt+nv+1:end]
    else            # whole subjects go to one set: 60/20/20 of the subjects
        G = unique(groups); gp = G[randperm(MersenneTwister(seed), length(G))]
        gt = floor(Int, 0.6length(G)); gv = floor(Int, 0.2length(G))
        rank = Dict(g => k for (k, g) in enumerate(gp)); r = randperm(MersenneTwister(seed + 3), N)
        part(lo, hi) = filter(i -> lo <= rank[groups[i]] <= hi, r)
        train, val, test = part(1, gt), part(gt + 1, gt + gv), part(gt + gv + 1, length(G))
    end
    trainval = vcat(train, val)
    t = time_ns()
    μ = mean(X[trainval, :]; dims=1); sd = std(X[trainval, :]; dims=1); sd[sd.==0] .= 1.0
    Xtv = (X[trainval, :] .- μ) ./ sd; prep_s = (time_ns() - t) * 1e-9
    Xte = (X[test, :] .- μ) ./ sd
    t = time_ns()
    Xs = size(Xtv, 1) > 20000 ? Xtv[randperm(MersenneTwister(seed + 2), size(Xtv, 1))[1:20000], :] : Xtv
    σ = median_distance(Xs); bandwidth_s = (time_ns() - t) * 1e-9
    rng = MersenneTwister(seed + 1); nt = length(train)
    t = time_ns(); li = stratified_landmarks(Y[train, :], m, rng); sample_train_s = (time_ns() - t) * 1e-9
    t = time_ns(); li1 = stratified_landmarks(Y[trainval, :], m, rng); sample_final_s = (time_ns() - t) * 1e-9
    t = time_ns()                                   # tuning stage: train rows only
    F = rbf(Xtv[1:nt, :], Xtv[li, :], σ); S = Matrix(Symmetric(F[li, :])); Fval = rbf(Xtv[nt+1:end, :], Xtv[li, :], σ)
    tune_s = (time_ns() - t) * 1e-9
    t = time_ns(); Ffinal = rbf(Xtv, Xtv[li1, :], σ); Sfinal = Matrix(Symmetric(Ffinal[li1, :])); final_s = (time_ns() - t) * 1e-9
    t = time_ns(); Ftest = rbf(Xte, Xtv[li1, :], σ); test_s = (time_ns() - t) * 1e-9
    (; F, S, Fval, Ffinal, Sfinal, Ftest, Xtv, Xte, σ, Ytrain=Y[train, :], Yval=Y[val, :], Ytv=Y[trainval, :], Ytest=Y[test, :],
       li,
       common_s=prep_s + bandwidth_s + sample_train_s + sample_final_s + tune_s + final_s + test_s)
end

"""Log-likelihood and accuracy of the coefficients B on (F, Y)."""
function loglik_accuracy(F, B, Y; full=false)
    η = F * B; full || (η = hcat(η, zeros(size(η, 1))))
    η .-= maximum(η; dims=2); P = exp.(η); P ./= sum(P; dims=2)
    sum(log(P[i, argmax(@view Y[i, :])]) for i in axes(Y, 1)),
    mean(argmax(@view P[i, :]) == argmax(@view Y[i, :]) for i in axes(Y, 1))
end

# Tuning path, λ choice and refit for "KMR_standard", "KMR_full" or "Newton".
function fit_real(d, method; tol=1e-2, path_cap=500, refit_cap=1000)
    full = method == "KMR_full"
    f = full ? KernelMultinomialAlg3.KernelMultinomial_full : KernelMultinomialAlg3.KernelMultinomial
    path(Y, F, S, λs, cap) = method == "Newton" ?
        newton_path("multinomial", F, S, Y, λs; tol=tol, measure=:maxabs, max_iters=cap) :
        (r = f(Y, F, S, λs; tol=tol, max_iters=cap); ([r[1][j, :, :] for j in axes(r[1], 1)], r[3]))
    start = time_ns()
    Bs, stats = path(d.Ytrain, d.F, d.S, LAMBDAS, path_cap)
    val = [loglik_accuracy(d.Fval, B, d.Yval; full=full)[1] for B in Bs]
    λ = LAMBDAS[argmax(val)]
    B, st = path(d.Ytv, d.Ffinal, d.Sfinal, [λ], refit_cap)
    ll, acc = loglik_accuracy(d.Ftest, B[1], d.Ytest; full=full)
    (method=method, lambda=λ, seconds=(time_ns() - start) * 1e-9 + d.common_s, test_loglik=ll, test_accuracy=acc,
     path_val_loglik=val, path_not_converged=count(s -> !s.converged, stats), refit_converged=st[1].converged)
end

"""Test accuracy of the 1-nearest-neighbor classifier (Euclidean distance on the
standardized features, fitted on train+validation)."""
function nearest_neighbor_accuracy(d)
    ytv = [argmax(r) for r in eachrow(d.Ytv)]; hits = 0; sq = vec(sum(abs2, d.Xtv; dims=2))
    for rows in Iterators.partition(axes(d.Xte, 1), 1024)
        D = sq' .- 2 .* (d.Xte[rows, :] * d.Xtv')          # ‖x‖² is constant per row
        for (k, i) in enumerate(rows)
            hits += ytv[argmin(@view D[k, :])] == argmax(@view d.Ytest[i, :])
        end
    end
    hits / size(d.Xte, 1)
end

const WARM = Ref(false)
# Compiles every step on a small random subset, so that no timing includes
# compilation.
function warmup_real(data)
    X, labels, _ = data; idx = randperm(MersenneTwister(0), size(X, 1))[1:1000]
    d = prepare_split(X[idx, :], labels[idx], nothing, 0, 30)
    for meth in ("KMR_standard", "KMR_full", "Newton"); fit_real(d, meth; path_cap=3, refit_cap=3); end
    nearest_neighbor_accuracy(d); nothing
end

"""
    real_split(dir, name, data, seed; ks, newton_ks)

Runs one split: for each k in `ks`, m = ⌊n/k⌋, QMME standard and full and,
for k in `newton_ks`, Newton's method; plus 1-NN once. Each (seed, m) is saved
in `dir` and skipped when it already exists.
"""
function real_split(dir, name, data, seed; ks=(64, 32, 16, 8, 4), newton_ks=(64, 32, 16))
    X, labels, groups = data
    for k in ks
        file = joinpath(dir, "$(name)_seed$(seed)_k$(k).jls"); isfile(file) && continue
        WARM[] || (warmup_real(data); WARM[] = true)
        m = projection_dim(size(X, 1), k); d = prepare_split(X, labels, groups, seed, m)
        rows = [fit_real(d, meth) for meth in (k in newton_ks ? ("KMR_standard", "KMR_full", "Newton") : ("KMR_standard", "KMR_full"))]
        knn = k == first(ks) ? nearest_neighbor_accuracy(d) : nothing
        save_result(file, (dataset=name, seed=seed, k=k, m=m, sigma=d.σ, rows=rows, knn_accuracy=knn))
        for r in rows
            @printf("%-5s seed %2d m=%-5d %-12s %8.2f s  test loglik %9.2f  accuracy %.4f\n",
                    name, seed, m, r.method, r.seconds, r.test_loglik, r.test_accuracy)
        end
        flush(stdout)
    end
end

"""Validation and test log-likelihood along the tuning path of one split
(standard parameterization), with stopping rule max|∇f| < tol (Figure 6)."""
function real_path_curve(name, data, seed, k; tol=1e-4, max_iters=10000)
    X, labels, groups = data
    d = prepare_split(X, labels, groups, seed, projection_dim(size(X, 1), k))
    Bs, _, stats = KernelMultinomialAlg3.KernelMultinomial(d.Ytrain, d.F, d.S, LAMBDAS; tol=tol, max_iters=max_iters)
    Ftest = rbf(d.Xte, d.Xtv[d.li, :], d.σ)             # test rows against the tuning landmarks
    all(s -> s.converged, stats) || @warn "$(count(s -> !s.converged, stats)) λ not converged"
    (lambda=collect(LAMBDAS), m=size(d.F, 2),
     val=[loglik_accuracy(d.Fval, Bs[j, :, :], d.Yval)[1] for j in axes(Bs, 1)],
     test=[loglik_accuracy(Ftest, Bs[j, :, :], d.Ytest)[1] for j in axes(Bs, 1)])
end
