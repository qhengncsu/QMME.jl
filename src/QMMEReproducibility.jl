"""
Code for the experiments of "Inertial Quadratic Majorization Minimization"
(Heng and Wang). The QMME solvers are in src/qmme; every figure and table is
produced by one script in scripts/ (see README.md).

Start Julia with `-t N`: the multinomial solvers use N threads, and BLAS is
set to N threads as well.
"""
module QMMEReproducibility

using LinearAlgebra, Random, Statistics, Serialization, Printf, Downloads
using Distributions, StatsBase, Optim, CSV, ZipFile, Plots

include("qmme/KernelQuantileQMME.jl")
include("qmme/KernelLogisticQMME.jl")
include("qmme/KernelMultinomialAlg2.jl")
include("qmme/KernelMultinomialAlg3.jl")
using .KernelQuantileQMME, .KernelLogisticQMME, .KernelMultinomialAlg2, .KernelMultinomialAlg3

include("utils.jl")
include("objectives.jl")
include("baselines.jl")
include("simulation.jl")
include("realdata.jl")

export KernelQuantileQMME, KernelLogisticQMME, KernelMultinomialAlg2, KernelMultinomialAlg3
export options, resultdir, cached, load_results, savefigure, savetable, meansd
export TAU, H_BW, objective, hessian, validation
export newton_path, newton_trace, fista_trace, adagd_trace, lbfgs_trace
export trajectory_instance, LAMBDAS, sim_path, sim_classes
export load_dataset, real_split, real_path_curve

__init__() = BLAS.set_num_threads(Threads.nthreads())

end
