module LogDensityProblemsExt

using ParallelMCMC
using LogDensityProblems: LogDensityProblems

"""
    DensityModel(ld; param_names=nothing)

Construct a `DensityModel` from any object implementing the
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface.

`ld` must support:
- `LogDensityProblems.capabilities(ld)` returning at least
  `LogDensityProblems.LogDensityOrder{1}` (i.e. gradient available).
- `LogDensityProblems.dimension(ld)` → `Int`
- `LogDensityProblems.logdensity_and_gradient(ld, x)` → `(logp, grad)`

The optional `param_names` keyword accepts a `Vector{Symbol}` of parameter names
that will be used for the columns of the returned `MCMCChains.Chains` object.
If omitted, names default to `x[1], x[2], ...` unless you also pass `param_names`
to `sample(...)`.

# Turing.jl / DynamicPPL example
```julia
using Turing, LogDensityProblems, LogDensityProblemsAD, Enzyme, ParallelMCMC, MCMCChains

@model function mymodel(y)
    μ ~ Normal(0, 1)
    y ~ Normal(μ, 0.5)
end

obs = 1.5
ld  = DynamicPPL.LogDensityFunction(mymodel(obs))
ldg = LogDensityProblemsAD.ADgradient(ADTypes.AutoEnzyme(), ld)

model = DensityModel(ldg; param_names=[:μ])
chain = sample(model, AdaptiveMALASampler(0.3; n_warmup=500), 2_000;
               chain_type=MCMCChains.Chains, discard_warmup=true, progress=true)
```

If both DynamicPPL and LogDensityProblemsAD are loaded, the simpler one-step
constructor `DensityModel(mymodel(obs))` is also available and extracts parameter
names automatically.
"""
function ParallelMCMC.DensityModel(ld; param_names=nothing)
    caps = LogDensityProblems.capabilities(ld)
    caps isa LogDensityProblems.LogDensityOrder{0} && error(
        "LogDensityProblems model must support gradients (LogDensityOrder{1} or higher). " *
        "Wrap it with LogDensityProblemsAD.ADgradient first.",
    )

    dim = LogDensityProblems.dimension(ld)

    logp(x) = LogDensityProblems.logdensity(ld, x)

    function gradlogp(x)
        _, g = LogDensityProblems.logdensity_and_gradient(ld, x)
        return g
    end

    return ParallelMCMC.DensityModel(logp, gradlogp, dim; param_names=param_names)
end

end # module
