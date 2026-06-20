module MCMCChainsExt

using MCMCChains: MCMCChains
using ParallelMCMC: ParallelMCMC
using AbstractMCMC: AbstractMCMC

ParallelMCMC.has_parallel_sample_implementation(::Type{MCMCChains.Chains}) = true

function ParallelMCMC._construct_chain(
    ::Type{MCMCChains.Chains},
    vals::AbstractMatrix{<:Real},
    internals::AbstractMatrix{<:Real},
    names::Vector{Symbol},
    internal_names::Vector{Symbol},
    model::ParallelMCMC.DensityModel,
    ::Union{Nothing,Vector},
)
    return MCMCChains.Chains(
        hcat(vals, internals),
        vcat(names, internal_names),
        Dict(:parameters => names, :internals => internal_names),
    )
end

end # module
