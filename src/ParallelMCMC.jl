module ParallelMCMC

using AbstractMCMC
using ADTypes: AbstractADType
using DifferentiationInterface: DifferentiationInterface
using FlexiChains
using LinearAlgebra
using OrderedCollections: OrderedDict
using Random
using Statistics

const DI = DifferentiationInterface

#=
Owned wrappers: identical semantics to their Base counterparts, but provide
stable function identities for backend-specific AD rules in `ext/EnzymeExt.jl`
without committing type piracy on `Base.*` / `Base.dot` / `Base.sum`. User
model code that wants those rules to fire (notably on GPU) should call these
instead. See `ext/EnzymeExt.jl` for the gc-transition abort they work around.

Why both a matmul and reductions: the GPU AD-HVP reverse path runs Enzyme
through `gradlogp` *and* through a scalar reduction wrapping it (see DEER's
`_HvpReverseClosure`). The reduction is what emits the `cuMemcpyDtoHAsync_v2`
that aborts Enzyme; owning both keeps each opaque to Enzyme's reverse rewriter.
=#
pmcmc_matmul(A::AbstractVecOrMat, B::AbstractVecOrMat) = A * B
pmcmc_dot(a::AbstractVector, b::AbstractVector) = dot(a, b)
pmcmc_dotsum(A::AbstractVecOrMat, B::AbstractVecOrMat) = sum(A .* B)

"""
    needs_host_staging(x::AbstractArray) -> Bool

Whether `x` has to be filled on the host and copied over, rather than written
element by element in place. `false` for anything with cheap scalar indexing,
which is the default.

The random fills — MALA's normal noise, DEER's Rademacher probes — are the only
places this matters: writing them one element at a time into device memory is
either an outright error or one kernel launch per element. Array types that
cannot take those writes say so here, and the fills stage through a host buffer
that the workspace keeps around.

Loading CUDA.jl opts `CuArray` in via `ext/CUDAExt.jl`; another device array
type is one method away.
"""
needs_host_staging(::AbstractArray) = false


include("MALA/MALA.jl")
include("DEER/DEERScan.jl")
include("DEER/DEER.jl")
include("interface.jl")

export DensityModel
export MALASampler, MALATransition, MALAState
export AdaptiveMALASampler, AdaptiveMALATransition, AdaptiveMALAState
export ParallelMALASampler, ParallelMALATransition, ParallelMALAState
export MALA, DEER
export pmcmc_matmul, pmcmc_dot, pmcmc_dotsum

# Re-exports for convenience
import AbstractMCMC: sample
export sample

end
