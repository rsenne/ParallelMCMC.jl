module DynamicPPLExt

using ParallelMCMC
using ADTypes: ADTypes
using DynamicPPL: DynamicPPL
using AbstractMCMC: AbstractMCMC
using Enzyme: Enzyme
using MCMCChains: MCMCChains
using LogDensityProblems: LogDensityProblems

"""
    DensityModel(turing_model::DynamicPPL.Model; ad_backend=ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Duplicated), hvp=nothing)

Convenience constructor: wraps a DynamicPPL/Turing `@model` directly as a
`DensityModel`, automatically extracting parameter names and wiring up gradient
computation via DynamicPPL's `adtype` interface.

Requires `DynamicPPL` to be loaded.

# Example
```julia
using Turing, ParallelMCMC, MCMCChains

@model function mymodel(y)
    μ ~ Normal(0, 1)
    y ~ Normal(μ, 0.5)
end

model = DensityModel(mymodel(1.5))
chain = sample(model, AdaptiveMALASampler(0.3; n_warmup=500), 2_000;
               chain_type=MCMCChains.Chains, discard_warmup=true, progress=true)
```
"""
function ParallelMCMC.DensityModel(
    turing_model::DynamicPPL.Model;
    ad_backend=ADTypes.AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Duplicated,
    ),
    hvp=nothing,
)
    # Sample in linked/unconstrained space and let DynamicPPL provide the gradient.
    ld = DynamicPPL.LogDensityFunction(
        turing_model,
        DynamicPPL.getlogjoint_internal,
        DynamicPPL.LinkAll();
        adtype=ad_backend,
    )
    # Requires LogDensityProblemsExt to be loaded
    return ParallelMCMC.DensityModel(ld; hvp=hvp)
end

# Types that represent LogDensityProblems objects that wrap DynamicPPL models.
const LDFPrimal = ParallelMCMC.LogDensityProblemPrimal{<:DynamicPPL.LogDensityFunction}
const LDFGradient = ParallelMCMC.LogDensityProblemGradient{<:DynamicPPL.LogDensityFunction}
const DensityModelLDF = ParallelMCMC.DensityModel{<:LDFPrimal,<:LDFGradient}

"""
    postprocess_sample(model::DensityModel, sample)

Converts a raw transition (e.g. `MALATransition`) into a `DynamicPPL.ParamsWithStats` object
by reevaluating the DynamicPPL model with the vectorised parameters. This requires that the
`DensityModel` object was constructed with a `LogDensityProblemPrimal` and
`LogDensityProblemGradient` that wrap a DynamicPPL model.
"""
function ParallelMCMC.postprocess_sample(
    model::DensityModelLDF, sample::ParallelMCMC.MALATransition
)
    stats = (accepted=sample.accepted,)
    return DynamicPPL.ParamsWithStats(sample.x, model.logdensity.ld, stats)
end
function ParallelMCMC.postprocess_sample(
    model::DensityModelLDF, sample::ParallelMCMC.ParallelMALATransition
)
    return DynamicPPL.ParamsWithStats(sample.x, model.logdensity.ld)
end
function ParallelMCMC.postprocess_sample(
    model::DensityModelLDF, sample::ParallelMCMC.AdaptiveMALATransition
)
    stats = (
        accepted=sample.accepted, step_size=sample.step_size, is_warmup=sample.is_warmup
    )
    return DynamicPPL.ParamsWithStats(sample.x, model.logdensity.ld, stats)
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:DynamicPPL.ParamsWithStats},
    model::DensityModel,
    spl::AbstractMCMC.AbstractSampler,
    state,
    chain_type::Type{MCMCChains.Chains};
    discard_warmup=false,
    kwargs...,
)
    if discard_warmup
        ts = filter(t -> !hasproperty(t.stats, :is_warmup) || !t.stats.is_warmup, ts)
    end
    return AbstractMCMC.from_samples(MCMCChains.Chains, hcat(ts))
end

function ParallelMCMC._construct_chain(
    ::Type{MCMCChains.Chains},
    vals::AbstractMatrix{Float64},
    internals::AbstractMatrix{Float64},
    names::Vector{Symbol},
    internal_names::Vector{Symbol},
    model::DensityModelLDF,
)
    pwss = map(zip(eachrow(vals), eachrow(internals))) do (val, internal)
        stats = NamedTuple{Tuple(internal_names)}(internal)
        DynamicPPL.ParamsWithStats(val, model.logdensity.ld, stats)
    end
    return AbstractMCMC.from_samples(MCMCChains.Chains, hcat(pwss))
end

end # module
