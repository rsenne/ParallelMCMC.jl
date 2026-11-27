module ParallelMCMC

using AbstractMCMC
using ADTypes: ADTypes, AbstractADType
using DifferentiationInterface: DifferentiationInterface
using FlexiChains
using LinearAlgebra
using OrderedCollections: OrderedDict
using Random
using ScopedValues: ScopedValue, with
using Statistics

const DI = DifferentiationInterface

# This wrapper lets EnzymeExt define a GPU differentiation rule without
# changing the rule for Julia's `*`. GPU models using forward-mode Enzyme
# should call it to avoid the compilation failure described in the GPU guide.
pmcmc_matmul(A::AbstractVecOrMat, B::AbstractVecOrMat) = A * B

"""
    needs_host_staging(x::AbstractArray) -> Bool

Whether `x` has to be filled on the host and copied over, rather than written
element by element in place. `false` for anything with cheap scalar indexing,
which is the default.

`ReactantExt` also reads it as "this array lives on a device", to warn when the
XLA client is on the host while the parameters are not.

`CUDAExt` defines it for `CuArray`. To support another device array type:

```julia
ParallelMCMC.needs_host_staging(::ROCArray) = true
```
"""
needs_host_staging(::AbstractArray) = false

"""
    _host_staging_buffer(template::AbstractArray, ::Type{T}, dims::Dims) -> Array{T}

Host buffer for staging copies between the host and arrays like `template`.
Device extensions can return pinned memory. Defaults to a plain `Array`.
"""
function _host_staging_buffer(::AbstractArray, ::Type{T}, dims::Dims) where {T}
    return Array{T}(undef, dims)
end

"""
    _copy_from_device_pointer!(dest::AbstractArray, ptr::Ptr{Cvoid}, platform::AbstractString) -> Bool

Copy device memory owned by another runtime into `dest`. `platform` is the
owner's XLA platform name ("cuda", "rocm", "cpu"). Returns `false` if `dest`'s
array type cannot address `ptr`, which is the default; on `true` the copy has
finished, so the caller may release the source.
"""
_copy_from_device_pointer!(::AbstractArray, ::Ptr{Cvoid}, ::AbstractString) = false

#= Lives here rather than in `DEER` because both DEER's `ReactantHVP` fallbacks
and `model.jl`'s gradient hooks report it. =#
const _REACTANT_LOAD_HINT = "AutoReactant requires Reactant.jl: add `using Reactant` to load ParallelMCMC's ReactantExt."

include("MALA/MALA.jl")
include("DEER/DEER.jl")
# Alias, not a second copy: the solver and CUDAExt use DEER.DEERScan.
const DEERScan = DEER.DEERScan
include("model.jl")
include("chains.jl")
include("samplers/mala.jl")
include("samplers/parallel_mala.jl")
include("samplers/adaptive_mala.jl")
include("sample.jl")

export DensityModel
export MALASampler, MALATransition, MALAState
export AdaptiveMALASampler, AdaptiveMALATransition, AdaptiveMALAState
export ParallelMALASampler, ParallelMALATransition, ParallelMALAState
export MALA, DEER
export pmcmc_matmul

# Re-exports for convenience
import AbstractMCMC: sample
export sample

end
