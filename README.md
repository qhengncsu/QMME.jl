# Reproducing "Inertial Quadratic Majorization Minimization"

Code for every figure and table in the experiments of Heng and Wang. The QMME
solvers are in `src/qmme/`; everything else (baselines, simulated and real
data, experiment drivers) is in `src/`, and each figure or table is produced
by one script in `scripts/`.

## Setup

Julia 1.11 (tested with 1.11.5). From this folder:

```
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -t 4 test/runtests.jl          # quick checks, about a minute
```

`Manifest.toml` pins every package version. Start Julia with `-t N`: the
multinomial solvers use N threads and BLAS is set to N threads. The timings
in the paper use `-t 4` on an Apple M4 (one experiment at a time).

The real datasets are downloaded from the UCI Machine Learning Repository on
first use and kept in `data/raw/`.

## Figures and tables

| Output | Script | Approximate time |
|---|---|---|
| Figures 1, 2 (`ObjTraj`, `GradNormTraj`) | `scripts/fig1_fig2_trajectories.jl` | 15 min |
| Figure 3 (`momentum`) | `scripts/fig3_momentum.jl` | 2 min |
| Figure 4 (`path`) | `scripts/fig4_solution_path.jl` | 1 day |
| Figure 5 (`increaseq`) | `scripts/fig5_increase_q.jl` | 5 h |
| Figure 6 (`realpath`) | `scripts/fig6_real_path.jl` | 15 min |
| Table 2 (Codon Usage) | `scripts/table_real_data.jl` | 7 h |
| Table 3 (HAR) | `scripts/table_real_data.jl --data har` | 3 h |

Run each as `julia --project -t 4 scripts/<script>.jl [options]`. Figures go
to `figures/` (pdf and png) and tables to `tables/`. The defaults give the
results in the paper; for example, `--reps 1` gives a quick version of
Figures 4, 5 and the tables.

## Settings

* QMME (Algorithm 1) resets the extrapolation counter when the objective at
  the extrapolated point increases or when the counter reaches the maximum
  restart period, set to 100.
* Section 4.1 (Figures 1–3): one instance, n = 2^14, m = 2^11, σ by the
  median heuristic, λ = 1e-4; all methods start from zero and stop at
  ‖∇f‖₂ < 1e-4 or after 1000 updates. Times exclude preprocessing (kernel
  matrices, factorizations, Lipschitz constants).
* Sections 4.2–4.3 (Figures 4, 5): 30 λ from 10 to 1e-5 with warm starts,
  ‖∇f‖₂ < 1e-4, at most 1000 iterations per λ; σ by the median heuristic on the
  training inputs; times include the kernel matrices.
* Section 5 (Figure 6, Tables 2, 3): 60/20/20 split (of the 30 subjects for
  HAR), σ by the median heuristic, class-stratified landmarks, max|∇f| < 1e-2
  with at most 500 iterations per λ on the path and 1000 for the refit; times
  include every step. Figure 6 uses max|∇f| < 1e-4 (at most 10000 iterations).

## Layout

```
src/qmme/          QMME solvers: quantile, logistic, multinomial (Algorithms 2 and 3)
src/objectives.jl  objectives, gradients and Hessians of the three problems
src/baselines.jl   Newton, FISTA, AdaGD, L-BFGS, linear models
src/simulation.jl  simulated data and the solution-path experiments
src/realdata.jl    Codon Usage and HAR: download, split, fit, evaluate
src/utils.jl       options, saved results, figure and table output
scripts/           one script per figure or table
test/runtests.jl   gradient/Hessian checks and QMME-versus-Newton agreement
```
