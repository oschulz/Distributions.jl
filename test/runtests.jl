using Distributions
using JSON, ForwardDiff, Calculus, PDMats # test dependencies
using Test
using Distributed
using Random
using StatsBase

tests = [
    "types",
    "utils",
    "samplers",
    "categorical",
    "univariates",
    "continuous",
    "fit",
    "multinomial",
    "binomial",
    "poissonbinomial",
    "dirichlet",
    "dirichletmultinomial",
    "mvnormal",
    "mvlognormal",
    "mvtdist",
    "kolmogorov",
    "edgeworth",
    "matrix",
    "vonmisesfisher",
    "conversion",
    "mixture",
    "gradlogpdf",
    "truncate",
    "noncentralt",
    "locationscale",
    "quantile_newton",
    "semicircle",
    "qq",
    "truncnormal",
]

printstyled("Running tests:\n", color=:blue)

if nworkers() > 1
    rmprocs(workers())
end

if Base.JLOptions().code_coverage == 1
    addprocs(1, exeflags = ["--code-coverage=user", "--inline=no", "--check-bounds=yes"])
else
    addprocs(1, exeflags = "--check-bounds=yes")
end

@everywhere using Random
@everywhere srand(345679)
res = pmap(tests) do t
    @eval module $(Symbol("Test_", t))
    using Distributions
    using JSON, ForwardDiff, Calculus, PDMats # test dependencies
    using Test
    using Random
    using LinearAlgebra
    using StatsBase
    include($t * ".jl")
    end
    return
end

# print method ambiguities
println("Potentially stale exports: ")
display(Test.detect_ambiguities(Distributions))
println()
