module CUDAExt

using ParallelMCMC: ParallelMCMC
using CUDA: CUDA, CuArray, CuPtr

ParallelMCMC.needs_host_staging(::CuArray) = true

# Pinned host memory: the driver can DMA directly instead of staging the copy.
function ParallelMCMC._host_staging_buffer(::CuArray, ::Type{T}, dims::Dims) where {T}
    return CUDA.pin(Array{T}(undef, dims))
end

# XLA and CUDA.jl share the device's primary context, so the pointer is usable
# as a `CuPtr`. The copy runs on CUDA.jl's stream, which XLA does not track, so
# synchronize before returning: the caller may release the source after this.
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
