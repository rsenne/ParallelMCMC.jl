module LogDensityProblemsExt

using ParallelMCMC
using LogDensityProblems: LogDensityProblems

"""
    DensityModel(ld; param_names=nothing, hvp=nothing)

Construct a `DensityModel` from any object implementing the
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface.

`ld` must support:
- `LogDensityProblems.capabilities(ld)` returning at least
  `LogDensityProblems.LogDensityOrder{1}` (i.e. gradient available).
- `LogDensityProblems.dimension(ld)` -> `Int`
- `LogDensityProblems.logdensity_and_gradient(ld, x)` -> `(logp, grad)`

The optional `param_names` keyword accepts a collection of parameter names that will be used
for the columns of the returned `FlexiChain` object. If omitted, a single vector-valued
parameter named `:x` will be chosen, unless you also pass `param_names` to `sample(...)`.

The `hvp` keyword argument is forwarded to the main `DensityModel` constructor.

# Turing.jl / DynamicPPL example
```julia
using Turing, LogDensityProblems, ADTypes, Enzyme, ParallelMCMC, FlexiChains

@model function mymodel(y)
    μ ~ Normal(0, 1)
    y ~ Normal(μ, 0.5)
end

obs = 1.5
ld = DynamicPPL.LogDensityFunction(
    mymodel(obs),
    DynamicPPL.getlogjoint_internal,
    DynamicPPL.LinkAll();
    adtype=ADTypes.AutoEnzyme(),
)

model = DensityModel(ld; param_names=[:μ])
chain = sample(model, AdaptiveMALASampler(0.3; n_warmup=500), 2_000;
               chain_type=VNChain, discard_warmup=true, progress=true)
```

If DynamicPPL is loaded, the simpler one-step constructor `DensityModel(mymodel(obs))`
is also available and extracts parameter names automatically.
"""
function ParallelMCMC.DensityModel(ld; param_names=nothing, hvp=nothing)
    caps = LogDensityProblems.capabilities(ld)
    caps isa LogDensityProblems.LogDensityOrder{0} && error(
        "LogDensityProblems model must support gradients (LogDensityOrder{1} or higher). " *
        "Construct it with gradient support enabled.",
    )

    dim = LogDensityProblems.dimension(ld)

    logp = ParallelMCMC.LogDensityProblemPrimal(ld)
    gradlogp = ParallelMCMC.LogDensityProblemGradient(ld)

    return ParallelMCMC.DensityModel(logp, gradlogp, dim; param_names=param_names, hvp=hvp)
end

(l::ParallelMCMC.LogDensityProblemPrimal)(x) = LogDensityProblems.logdensity(l.ld, x)
function (l::ParallelMCMC.LogDensityProblemGradient)(x)
    return last(LogDensityProblems.logdensity_and_gradient(l.ld, x))
end

end # module
