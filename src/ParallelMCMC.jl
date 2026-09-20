module ParallelMCMC

using AbstractMCMC
using ADTypes: ADTypes, AbstractADType
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
=#
pmcmc_matmul(A::AbstractVecOrMat, B::AbstractVecOrMat) = A * B
pmcmc_dot(a::AbstractVector, b::AbstractVector) = dot(a, b)
pmcmc_dotsum(A::AbstractVecOrMat, B::AbstractVecOrMat) = sum(A .* B)

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

Host buffer for staging transfers to and from arrays like `template`. Device
extensions can return pinned memory. Defaults to a plain `Array`.
"""
function _host_staging_buffer(::AbstractArray, ::Type{T}, dims::Dims) where {T}
    return Array{T}(undef, dims)
end

"""
    _device_array_from_pointer(template::AbstractArray, ::Type{T}, ptr::Ptr{Cvoid}, dims::Dims, platform::AbstractString)

Wrap device memory owned by another runtime as an array like `template`, or
return `nothing` if that array type cannot address it. `platform` is the owner's
XLA platform name (`"cuda"`, `"rocm"`, `"cpu"`). The result aliases `ptr` and
does not own it; copy out of it while the owner still holds the memory.
Defaults to `nothing`.
"""
function _device_array_from_pointer(
    ::AbstractArray, ::Type, ::Ptr{Cvoid}, ::Dims, ::AbstractString
)
    return nothing
end

#= Lives here rather than in `DEER` because both DEER's `ReactantHVP` fallbacks
and `interface.jl`'s gradient hooks report it. =#
const _REACTANT_LOAD_HINT = "AutoReactant requires Reactant.jl: add `using Reactant` to load ParallelMCMC's ReactantExt."

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
