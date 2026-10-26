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
