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

######################
# Chain construction #
######################
# In this section, we define overloads for DynamicPPL-based models so that resulting chains
# are converted back into the original parameter space and contain the correct parameter
# names. This is done by converting the raw samples (vectors of parameters) back into
# `DynamicPPL.ParamsWithStats` objects.

const ParallelMCMCTransitionTypes = Union{
    ParallelMCMC.MALATransition,
    ParallelMCMC.AdaptiveMALATransition,
    ParallelMCMC.ParallelMALATransition,
}
# Types that represent LogDensityProblems objects that wrap DynamicPPL models.
const LDFPrimal = ParallelMCMC.LogDensityProblemPrimal{<:DynamicPPL.LogDensityFunction}
const LDFGradient = ParallelMCMC.LogDensityProblemGradient{<:DynamicPPL.LogDensityFunction}
const DensityModelLDF = ParallelMCMC.DensityModel{<:LDFPrimal,<:LDFGradient}

function AbstractMCMC.bundle_samples(
    ts::Vector{<:ParallelMCMC.MALATransition},
    model::DensityModelLDF,
    spl::ParallelMCMC.MALASampler,
    state::ParallelMCMC.MALAState,
    chain_type::Type{MCMCChains.Chains};
    kwargs...,
)
    return make_processed_dynamicppl_chain(MCMCChains.Chains, ts, model)
end
function AbstractMCMC.bundle_samples(
    ts::Vector{<:ParallelMCMC.ParallelMALATransition},
    model::DensityModelLDF,
    spl::ParallelMCMC.ParallelMALASampler,
    state::ParallelMCMC.ParallelMALAState,
    chain_type::Type{MCMCChains.Chains};
    kwargs...,
)
    return make_processed_dynamicppl_chain(MCMCChains.Chains, ts, model)
end
function AbstractMCMC.bundle_samples(
    ts::Vector{<:ParallelMCMC.AdaptiveMALATransition},
    model::DensityModelLDF,
    spl::ParallelMCMC.AdaptiveMALASampler,
    state::ParallelMCMC.AdaptiveMALAState,
    chain_type::Type{MCMCChains.Chains};
    discard_warmup::Bool=false,
    kwargs...,
)
    ts = discard_warmup ? filter(t -> !t.is_warmup, ts) : ts
    return make_processed_dynamicppl_chain(MCMCChains.Chains, ts, model)
end

"""
    getstats(sample::ParallelMCMCTransitionTypes)

Get a `NamedTuple` of stats from an MCMC transition.
"""
getstats(sample::ParallelMCMC.MALATransition) = (accepted=sample.accepted,)
function getstats(sample::ParallelMCMC.AdaptiveMALATransition)
    return (
        accepted=sample.accepted, step_size=sample.step_size, is_warmup=sample.is_warmup
    )
end
getstats(::ParallelMCMCTransitionTypes) = (;)

function make_processed_dynamicppl_chain(
    ::Type{Tchain}, ts::Vector{<:ParallelMCMCTransitionTypes}, model::DensityModelLDF
) where {Tchain}
    pwss = map(ts) do t
        # Note: This assumes that there is always a field called t.x. This is currently true
        # of all samplers in ParallelMCMC
        DynamicPPL.ParamsWithStats(t.x, model.logdensity.ld, getstats(t))
    end
    return AbstractMCMC.from_samples(Tchain, hcat(pwss))
end

function ParallelMCMC._construct_chain(
    ::Type{MCMCChains.Chains},
    vals::AbstractMatrix{<:Real},
    internals::AbstractMatrix{<:Real},
    ::Vector{Symbol},
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
