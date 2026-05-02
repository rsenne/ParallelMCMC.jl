module ParallelMCMCBenchmarks

export ParallelMCMCPrBenchmarks

include("models/bayes_linreg.jl")
include("models/bayes_logreg.jl")
include("runners/mala_runner.jl")
include("pr_suite.jl")

end # module
