module CUDAExt

using ParallelMCMC: ParallelMCMC
using CUDA: CUDA, CuArray, CuPtr

ParallelMCMC.needs_host_staging(::CuArray) = true

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
