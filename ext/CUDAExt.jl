module CUDAExt

using ParallelMCMC: ParallelMCMC
using CUDA: CUDA, CuArray, CuPtr

ParallelMCMC.needs_host_staging(::CuArray) = true

# Pinned host memory: the driver can DMA directly instead of staging the copy.
function ParallelMCMC._host_staging_buffer(::CuArray, ::Type{T}, dims::Dims) where {T}
    return CUDA.pin(Array{T}(undef, dims))
end

# Non-owning view of an XLA buffer. XLA and CUDA.jl share the device's primary
# context, so the pointer is usable as a `CuPtr`. Callers copy out of the view
# while XLA still holds the buffer.
function ParallelMCMC._device_array_from_pointer(
    ::CuArray, ::Type{T}, ptr::Ptr{Cvoid}, dims::Dims, platform::AbstractString
) where {T}
    platform == "cuda" || return nothing
    return unsafe_wrap(CuArray, reinterpret(CuPtr{T}, UInt(ptr)), dims; own=false)
end

end
