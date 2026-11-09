#= Sequential MALA references the tests compare DEER and the batched kernels
against. They are built from the kernel functions in `ParallelMCMC.MALA`; the
package itself never calls them. Include with

    isdefined(@__MODULE__, :TestUtils) || include("utils.jl")
    using .TestUtils
=#
module TestUtils

using ParallelMCMC: MALA, AbstractMCMC

export run_mala_sequential_taped, mala_accept_indicator, mala_step_full
export step_into_first_block

"""
    step_into_first_block(rng, model, sampler; kwargs...) -> (transition, state)

The initial step of a `ParallelMALASampler` returns the initial point and solves
nothing; the step after it solves the first DEER block. Returns that second step.
"""
function step_into_first_block(rng, model, sampler; kwargs...)
    _, state = AbstractMCMC.step(rng, model, sampler; kwargs...)
    return AbstractMCMC.step(rng, model, sampler, state)
end

"""
Run sequential taped MALA for T steps.

Returns:
- xs: Vector of states length T+1, xs[1]=x0, xs[t+1]=x_t
"""
function run_mala_sequential_taped(
    logp,
    gradlogp,
    x0::AbstractVector,
    ϵ::Real,
    ξs::Vector{<:AbstractVector},
    us::AbstractVector;
    cholM=nothing,
)
    T = length(us)
    length(ξs) == T || throw(DimensionMismatch("ξs and us must have the same length"))
    xs = Vector{typeof(x0)}(undef, T + 1)
    xs[1] = copy(x0)

    ws = MALA.MALAWorkspace(x0)
    x = copy(x0)
    x_next = similar(x0)

    for t in 1:T
        MALA.mala_step_taped!(x_next, ws, logp, gradlogp, x, ϵ, ξs[t], us[t]; cholM=cholM)
        xs[t + 1] = copy(x_next)
        x, x_next = x_next, x
    end
    return xs
end

"""
Primal accept indicator for a taped MALA step.
Returns a float in {0, 1} matching the precision of `u`.
"""
function mala_accept_indicator(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real; cholM=nothing
)
    ws = MALA.MALAWorkspace(x)
    MALA.mala_proposal!(ws.y, ws, logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    logα = MALA.mala_logα!(ws, logp, gradlogp, x, ws.y, ϵ; cholM=cholM)
    FP = typeof(float(u))
    return (log(u) < logα) ? one(FP) : zero(FP)
end

function mala_step_full(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real; cholM=nothing
)
    x_next, accepted, _ = MALA.mala_step_with_logα(logp, gradlogp, x, ϵ, ξ, u; cholM=cholM)
    return x_next, accepted
end

end # module
