# QMME.jl

This repository contains the code for the paper

> Qiang Heng and Caixing Wang. *Inertial Quadratic Majorization Minimization
> with Application to Kernel Regularized Learning.*

It includes the QMME algorithms and one script for each figure and table in the
paper.

## Installation

1. Install [Julia](https://julialang.org/downloads/) 1.11.
2. Download this repository and enter its folder:
   ```
   git clone https://github.com/qhengncsu/QMME.jl.git
   cd QMME.jl
   ```
3. Install the required packages. `Manifest.toml` fixes their versions, so you
   get the same versions we used:
   ```
   julia --project -e 'using Pkg; Pkg.instantiate()'
   ```
4. Optionally, check that everything works (about a minute):
   ```
   julia --project -t 4 test/runtests.jl
   ```

## Reproducing the figures and tables

Each figure or table is produced by one script. For example, to produce
Figure 3, run
```
julia --project -t 4 scripts/fig3_momentum.jl
```
and the figure is saved as `figures/momentum.pdf`. The option `-t 4` uses four
threads, as in the paper.

| Output | Script | Approximate time |
|---|---|---|
| Figures 1, 2 (`ObjTraj`, `GradNormTraj`) | `scripts/fig1_fig2_trajectories.jl` | 15 min |
| Figure 3 (`momentum`) | `scripts/fig3_momentum.jl` | 2 min |
| Figure 4 (`path`) | `scripts/fig4_solution_path.jl` | 1 day |
| Figure 5 (`increaseq`) | `scripts/fig5_increase_q.jl` | 5 h |
| Figure 6 (`realpath`) | `scripts/fig6_real_path.jl` | 15 min |
| Table 2 (Codon Usage) | `scripts/table_real_data.jl` | 7 h |
| Table 3 (HAR) | `scripts/table_real_data.jl --data har` | 3 h |

Figures are saved in `figures/` (pdf and png) and tables in `tables/`. The two
real datasets, Codon Usage and HAR, are downloaded from the UCI Machine
Learning Repository the first time they are needed.

Figures 4 and 5 and the two tables average over 20 replications, so they take
hours. To get a quick version first, run a single replication, for example
```
julia --project -t 4 scripts/fig4_solution_path.jl --reps 1 --newton-reps 1
```

### Additional solution-path comparisons

Run these commands from the repository root to reproduce the Section 4.2
comparison and Table R1 in the response. The original path timings were
collected with Julia 1.12.7 and four threads. The first command runs L-BFGS,
QMME and Newton on the same five instances. The second adds Newton–CG on the
first instance for Table R1.

```sh
QMME_REPS=5 julia --project=. -t4 scripts/competitive_paths.jl
QMME_METHODS=Newton-PCG QMME_REPS=1 julia --project=. -t4 scripts/competitive_paths.jl
python3 scripts/summarize_competitive_paths.py
```

Each run uses 30 warm-started penalties at both projection dimensions
(128 and 512). L-BFGS uses memory 5 and up to 1,000 updates per penalty.
Newton–CG uses matrix-free Hessian products and a block-Jacobi preconditioner
with blocks of up to 64 coefficients within each class. For the multinomial
comparison at m = 512, the summary averages the five path times (kernel
construction plus solver time) and counts how many of the 150 fits have a
gradient norm below 1e-4. It also reports L-BFGS results for each model and
dimension, and the single-instance results for Table R1.
Per-penalty records are saved in `results/competitive_paths/`;
completed paths are reused. Set `QMME_OUT` to a fresh directory when
changing settings, and pass that directory to the summary script.

To check Hessian products, preconditioned CG residuals and convergence on
small problems, run `julia --project=. -t4 test/validate_path_baselines.jl`.
