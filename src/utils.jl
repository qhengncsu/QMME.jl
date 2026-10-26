# Command-line options, saved results and the plotting style shared by the
# scripts in scripts/.

const ROOT = dirname(@__DIR__)
resultdir(parts...) = mkpath(joinpath(ROOT, "results", parts...))
const FIGDIR = joinpath(ROOT, "figures")
const TABDIR = joinpath(ROOT, "tables")

"""
    options(defaults::NamedTuple)

Command-line options `--name value` (dashes or underscores) overriding
`defaults`; values are parsed to the type of the default.
"""
function options(defaults::NamedTuple; args=ARGS)
    d = Dict{Symbol,Any}(pairs(defaults)); i = 1
    while i <= length(args)
        k = Symbol(replace(lstrip(args[i], '-'), "-" => "_"))
        haskey(d, k) && i < length(args) || error("unknown option $(args[i]); options: " *
            join(("--$(replace(string(k), "_" => "-")) (default $v)" for (k, v) in pairs(defaults)), ", "))
        T = typeof(defaults[k]); d[k] = T <: AbstractString ? args[i+1] : parse(T, args[i+1]); i += 2
    end
    NamedTuple{keys(defaults)}(Tuple(d[k] for k in keys(defaults)))
end

"""Writes `x` to `file` atomically (a partial file never looks like a result)."""
function save_result(file, x)
    tmp = file * ".tmp$(getpid())"; serialize(tmp, x); mv(tmp, file; force=true); x
end

"""Value saved in `file`, computing and saving it with `f()` the first time."""
cached(f, file) = isfile(file) ? deserialize(file) : save_result(file, f())

"""All results saved in `dir` (every .jls file)."""
load_results(dir) = [deserialize(joinpath(dir, f)) for f in sort(readdir(dir)) if endswith(f, ".jls")]

"""Saves the current plot as figures/<name>.pdf and .png."""
function savefigure(name)
    mkpath(FIGDIR)
    for ext in (".pdf", ".png"); savefig(joinpath(FIGDIR, name * ext)); end
    println("Wrote ", joinpath(FIGDIR, name * ".pdf"))
end

"""Writes a LaTeX table to tables/<name>.tex and prints it."""
function savetable(name, lines)
    mkpath(TABDIR); s = join(lines, "\n") * "\n"
    write(joinpath(TABDIR, name * ".tex"), s); print(s)
    println("Wrote ", joinpath(TABDIR, name * ".tex"))
end

"Mean (standard deviation), or the mean alone for a single value."
meansd(x) = length(x) > 1 ? @sprintf("%.2f (%.2f)", mean(x), std(x)) : @sprintf("%.2f", only(x))
