module DynamicPPLExt

using ParallelMCMC
using ADTypes: ADTypes
using DynamicPPL: DynamicPPL
using AbstractMCMC: AbstractMCMC
using FlexiChains: FlexiChain, VarName
using MCMCChains: MCMCChains
using LogDensityProblems: LogDensityProblems

CHAIN_TYPES = (MCMCChains.Chains, FlexiChain{VarName})

"""
    DensityModel(turing_model::DynamicPPL.Model; ad_backend=ADTypes.AutoForwardDiff(), hvp=nothing)

Convenience constructor: wraps a DynamicPPL/Turing `@model` directly as a
`DensityModel`, automatically extracting parameter names and wiring up gradient
computation via DynamicPPL's `adtype` interface.

Requires `DynamicPPL`, `ForwardDiff`, and `LogDensityProblems` to be loaded (these are the
weak-dependency triggers for this extension; `ForwardDiff` is what backs the default
`AutoForwardDiff()` AD path).

# Example
```julia
using Turing, ParallelMCMC, MCMCChains

@model function mymodel(y)
    μ ~ Normal(0, 1)
    y ~ Normal(μ, 0.5)
end

# AutoForwardDiff is the default. For larger models pass an explicit backend
# and `using` the corresponding package (Enzyme, Mooncake).
model = DensityModel(mymodel(1.5))
chain = sample(model, AdaptiveMALASampler(0.3; n_warmup=500), 2_000;
               chain_type=MCMCChains.Chains, discard_warmup=true, progress=true)
```
"""
function ParallelMCMC.DensityModel(
    turing_model::DynamicPPL.Model; ad_backend=ADTypes.AutoForwardDiff(), hvp=nothing
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

"""
    is_warmup(sample::ParallelMCMCTransitionTypes)

Check if a sample is from the warmup phase of MCMC sampling.
"""
is_warmup(::ParallelMCMCTransitionTypes) = false
is_warmup(sample::ParallelMCMC.AdaptiveMALATransition) = sample.is_warmup

for (Ttrans, Tspl, Tstate) in (
    (ParallelMCMC.MALATransition, ParallelMCMC.MALASampler, ParallelMCMC.MALAState),
    (
        ParallelMCMC.ParallelMALATransition,
        ParallelMCMC.ParallelMALASampler,
        ParallelMCMC.ParallelMALAState,
    ),
    (
        ParallelMCMC.AdaptiveMALATransition,
        ParallelMCMC.AdaptiveMALASampler,
        ParallelMCMC.AdaptiveMALAState,
    ),
)
    for Tchain in CHAIN_TYPES
        @eval begin
            function AbstractMCMC.bundle_samples(
                ts::Vector{<:$Ttrans},
                model::DensityModelLDF,
                spl::$Tspl,
                state::$Tstate,
                chain_type::Type{$Tchain},
                discard_warmup::Bool=false,
                kwargs...,
            )
                ts = discard_warmup ? filter(t -> !is_warmup(t), ts) : ts
                return make_processed_dynamicppl_chain($Tchain, ts, model)
            end
        end
    end
end

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

# Must define separate methods for each chain type to avoid method ambiguity
for T in CHAIN_TYPES
    @eval function ParallelMCMC._construct_chain(
        ::Type{$T},
        vals::AbstractMatrix{<:Real},
        internals::AbstractMatrix{<:Real},
        ::Vector{Symbol},
        internal_names::Vector{Symbol},
        model::DensityModelLDF,
        ::Union{Nothing,Vector}
    )
        pwss = map(zip(eachrow(vals), eachrow(internals))) do (val, internal)
            stats = NamedTuple{Tuple(internal_names)}(internal)
            DynamicPPL.ParamsWithStats(val, model.logdensity.ld, stats)
        end
        return AbstractMCMC.from_samples($T, hcat(pwss))
    end
end

end # module
