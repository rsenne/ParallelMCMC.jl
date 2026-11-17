#= `sample` on a `DensityModel` returns a `VNChain` unless `chain_type` says
otherwise. AbstractMCMC's own default is `Any`, which hands back the raw
transition vector; pass `chain_type=Any` to get that. =#
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::Union{MALASampler,AdaptiveMALASampler,ParallelMALASampler},
    N_or_isdone;
    chain_type::Type=VNChain,
    kwargs...,
)
    return AbstractMCMC.mcmcsample(rng, model, sampler, N_or_isdone; chain_type, kwargs...)
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::Union{MALASampler,AdaptiveMALASampler,ParallelMALASampler},
    parallel::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    chain_type::Type=VNChain,
    kwargs...,
)
    cache = sampler isa ParallelMALASampler ? SampleCache() : nothing
    return with(SAMPLE_CACHE => cache) do
        return AbstractMCMC.mcmcsample(
            rng, model, sampler, parallel, N, nchains; chain_type, kwargs...
        )
    end
end
