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
synchronize before returning, since the caller may release the source. =#
function ParallelMCMC._copy_from_device_pointer!(
    dest::CuArray{T}, ptr::Ptr{Cvoid}, platform::AbstractString
) where {T}
    platform == "cuda" || return false
    src = unsafe_wrap(CuArray, reinterpret(CuPtr{T}, UInt(ptr)), size(dest); own=false)
    copyto!(dest, src)
    CUDA.synchronize()
    return true
end

end
