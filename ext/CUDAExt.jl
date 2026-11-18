module CUDAExt

using ParallelMCMC: ParallelMCMC
using CUDA: CUDA, CuArray, CuMatrix, CuPtr
using Random: Random, AbstractRNG

const DEER = ParallelMCMC.DEER

#= The scan and convergence-check kernels are KernelAbstractions kernels in the core
package, and `needs_host_staging` covers every GPU array there too. What remains
here needs CUDA itself: its RNG, pinned memory and raw device pointers. =#

#= Draw the probes on the device, seeded from the caller's `rng`. A solve stays
reproducible, but its probes differ from the host path's, so DEER takes a
different route to the same trajectory. =#
DEER._device_probe_rng(::CuArray) = CUDA.RNG()

function DEER._probe_matrix!(
    Z::CuMatrix{T}, rng::AbstractRNG, _, dev::CUDA.RNG
) where {T<:AbstractFloat}
    Random.seed!(dev, rand(rng, UInt32))
    Random.rand!(dev, Z)
    Z .= ifelse.(Z .< T(0.5), -one(T), one(T))
    return Z
end

#= Pinned, so the device-to-host copy into this buffer can DMA directly.
`CUDA.pin` returns `nothing` for an already-registered address, so don't
forward its result. =#
function ParallelMCMC._host_staging_buffer(::CuArray, ::Type{T}, dims::Dims) where {T}
    buf = Array{T}(undef, dims)
    CUDA.pin(buf)
    return buf
end

#= XLA and CUDA.jl share the device's primary context, so the pointer is usable
as a `CuPtr`. The copy runs on CUDA.jl's stream, which XLA does not track;
synchronize before returning, since the caller may release the source.

On a multi-GPU machine the XLA buffer can sit on a different device than
`dest`. Ask the driver which device owns the pointer and decline on a
mismatch=#
function ParallelMCMC._copy_from_device_pointer!(
    dest::CuArray{T}, ptr::Ptr{Cvoid}, platform::AbstractString
) where {T}
    platform == "cuda" || return false
    src_ptr = reinterpret(CuPtr{T}, UInt(ptr))
    dev = CUDA.device(dest)
    CUDA.device(src_ptr) == dev || return false
    CUDA.device!(dev) do
        src = unsafe_wrap(CuArray, src_ptr, size(dest); own=false)
        copyto!(dest, src)
        return CUDA.synchronize()
    end
    return true
end

end
