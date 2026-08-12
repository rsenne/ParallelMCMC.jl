module LogDensityProblemsExt

using ParallelMCMC
using LogDensityProblems: LogDensityProblems

"""
    DensityModel(ld; param_names=nothing, hvp=nothing,
                 logdensity_batch=nothing, grad_logdensity_batch=nothing, hvp_batch=nothing)

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

`hvp` and the batched slots are forwarded to the main `DensityModel` constructor
and keep their meaning there, with one caveat. `ld` fills the gradient slot, and
a gradient `ld` computes by AD carries a preparation tied to its input type, so
it rejects the tangents a plain `hvp` backend would push through it. Use a
callable, or a `DifferentiationInterface.SecondOrder`, which differentiates the
log-density instead. Same for `hvp_batch`. `ADTypes.AutoReactant()` is out here
too, DI or no DI: Reactant cannot trace the DynamicPPL/LogDensityProblems
machinery `ld`'s gradient dispatches into.

`ld` supplies no batched log-density, so reaching the batched DEER path means
writing `logdensity_batch` by hand.

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
function ParallelMCMC.DensityModel(
    ld;
    param_names=nothing,
    hvp=nothing,
    logdensity_batch=nothing,
    grad_logdensity_batch=nothing,
    hvp_batch=nothing,
)
    caps = LogDensityProblems.capabilities(ld)
    caps isa LogDensityProblems.LogDensityOrder{0} && error(
        "LogDensityProblems model must support gradients (LogDensityOrder{1} or higher). " *
        "Construct it with gradient support enabled.",
    )

    dim = LogDensityProblems.dimension(ld)

    logp = ParallelMCMC.LogDensityProblemPrimal(ld)
    gradlogp = ParallelMCMC.LogDensityProblemGradient(ld)

    return ParallelMCMC.DensityModel(
        logp,
        gradlogp,
        dim;
        param_names=param_names,
        hvp=hvp,
        logdensity_batch=logdensity_batch,
        grad_logdensity_batch=grad_logdensity_batch,
        hvp_batch=hvp_batch,
    )
end

(l::ParallelMCMC.LogDensityProblemPrimal)(x) = LogDensityProblems.logdensity(l.ld, x)
function (l::ParallelMCMC.LogDensityProblemGradient)(x)
    return last(LogDensityProblems.logdensity_and_gradient(l.ld, x))
end

end # module
