module CUDAExt

using ParallelMCMC: ParallelMCMC
using CUDA: CUDA, CuArray, CuMatrix, CuVector, CuPtr, @cuda
using Random: Random, AbstractRNG

const DEER = ParallelMCMC.DEER
const DEERScan = DEER.DEERScan

ParallelMCMC.needs_host_staging(::CuArray) = true

#= On some platforms, notably Windows, a kernel launch costs more than a small
kernel's work. These methods cut the launches in the affine scan, the
convergence test, and the Hutchinson probes. =#

#= Chunked scan of s_t = A[:, t] .* s_{t-1} .+ B[:, t] in three launches. Thread
(d, k) composes the maps of chunk k for parameter d, one thread per d carries the
state across chunks, and thread (d, k) replays its chunk from that state. Each
state comes from the same sequential recursion as `solve_affine_seq!`. =#
function _chunk_summary_kernel!(Asum, Bsum, A, B, C::Int, T::Int)
    D = size(A, 1)
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    i > D * size(Asum, 2) && return nothing
    d = (i - 1) % D + 1
    k = (i - 1) ÷ D + 1
    a = one(eltype(A))
    b = zero(eltype(B))
    @inbounds for t in ((k - 1) * C + 1):min(k * C, T)
        at = A[d, t]
        a = at * a
        b = at * b + B[d, t]
    end
    @inbounds Asum[d, k] = a
    @inbounds Bsum[d, k] = b
    return nothing
end

function _chunk_carry_kernel!(Sin, Asum, Bsum, s0)
    d = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    d > length(s0) && return nothing
    @inbounds s = s0[d]
    @inbounds for k in axes(Sin, 2)
        Sin[d, k] = s
        s = Asum[d, k] * s + Bsum[d, k]
    end
    return nothing
end

function _chunk_replay_kernel!(S, A, B, Sin, C::Int, T::Int)
    D = size(A, 1)
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    i > D * size(Sin, 2) && return nothing
    d = (i - 1) % D + 1
    k = (i - 1) ÷ D + 1
    @inbounds s = Sin[d, k]
    @inbounds for t in ((k - 1) * C + 1):min(k * C, T)
        s = A[d, t] * s + B[d, t]
        S[d, t] = s
    end
    return nothing
end

function DEERScan.solve_affine_scan_diag!(
    S::CuMatrix{T},
    A::CuMatrix{T},
    B::CuMatrix{T},
    s0::CuVector{T},
    ws::DEERScan.AffineScanWorkspace;
    check_shapes::Bool=true,
) where {T}
    D, Tn = size(A)
    #= The kernels index with `@inbounds`, so the sizes are checked even when
    `check_shapes=false`; with mismatched sizes they would read and write out of
    bounds instead of throwing. =#
    DEERScan._check_scan_shapes(S, A, B, s0, ws)
    (Tn == 0 || D == 0) && return S
    #= C ≈ √T balances the sequential work in the carry kernel (K steps) against
    that in the other two (C steps each). =#
    C = max(32, isqrt(Tn))
    K = cld(Tn, C)
    Asum = view(ws.alpha, :, 1:K)
    Bsum = view(ws.beta, :, 1:K)
    Sin = view(ws.alpha_new, :, 1:K)
    threads = 256
    blocks = cld(D * K, threads)
    @cuda threads = threads blocks = blocks _chunk_summary_kernel!(Asum, Bsum, A, B, C, Tn)
    @cuda threads = min(D, threads) blocks = cld(D, threads) _chunk_carry_kernel!(
        Sin, Asum, Bsum, s0
    )
    @cuda threads = threads blocks = blocks _chunk_replay_kernel!(S, A, B, Sin, C, Tn)
    return S
end

#= Both maxima of the DEER convergence test in one kernel and one 2-element copy
to the host. Each warp reduces its values with shuffles and its first lane
updates the result with an atomic max on the bit patterns, which order the same
way as the values for non-negative floats. A NaN has larger bits than any
finite value or Inf, so it propagates as it does with `maximum`. =#
_bits(::Type{Float32}) = UInt32
_bits(::Type{Float64}) = UInt64

function _max_abs_change_kernel!(out, S_new, S)
    T = eltype(S_new)
    n = length(S_new)
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    dmax = zero(T)
    smax = zero(T)
    while i <= n
        @inbounds a = S_new[i]
        @inbounds dmax = max(dmax, abs(a - S[i]))
        smax = max(smax, abs(a))
        i += stride
    end
    offset = 16
    while offset > 0
        dmax = max(dmax, CUDA.shfl_down_sync(0xffffffff, dmax, offset))
        smax = max(smax, CUDA.shfl_down_sync(0xffffffff, smax, offset))
        offset >>= 1
    end
    if CUDA.laneid() == 1
        CUDA.atomic_max!(pointer(out, 1), reinterpret(_bits(T), dmax))
        CUDA.atomic_max!(pointer(out, 2), reinterpret(_bits(T), smax))
    end
    return nothing
end

function DEER._max_abs_change(
    S_new::CuArray{T}, S::CuArray{T}
) where {T<:Union{Float32,Float64}}
    # The kernel reads `S[i]` with `@inbounds` for every index of `S_new`.
    size(S_new) == size(S) ||
        throw(DimensionMismatch("S_new and S must have the same size"))
    out = CUDA.zeros(_bits(T), 2)
    threads = 256
    blocks = min(cld(length(S_new), threads), 1024)
    @cuda threads = threads blocks = max(blocks, 1) _max_abs_change_kernel!(out, S_new, S)
    h = Array(out)
    return reinterpret(T, h[1]), reinterpret(T, h[2])
end

#= Probes on the device. The seed comes from the caller's `rng`, so a solve is
reproducible, but the probes differ from those of the host path. That changes
only how DEER reaches the trajectory, not the trajectory it converges to. =#
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
