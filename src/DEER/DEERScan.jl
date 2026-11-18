module DEERScan

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const
import ...ParallelMCMC: _kernel_backend

export AffineScanWorkspace,
    solve_affine_seq!,
    solve_affine_seq,
    solve_affine_scan_diag!,
    solve_affine_scan_diag,
    check_affine_scan,
    affine_scan_residual

"""
Reusable workspace for the diagonal affine scan.

Buffers are allocated with the same array type / device placement as the template
matrix used to construct the workspace, so this works on GPU arrays and CPU
arrays alike.
"""
struct AffineScanWorkspace{M}
    alpha::M
    beta::M
    alpha_new::M
    beta_new::M
end

function AffineScanWorkspace(A_template::AbstractMatrix)
    alpha = similar(A_template)
    beta = similar(A_template)
    alpha_new = similar(A_template)
    beta_new = similar(A_template)
    return AffineScanWorkspace(alpha, beta, alpha_new, beta_new)
end

"""
    solve_affine_seq!(S, A, B, s0)

Reference sequential solver for the diagonal affine recurrence

    s_t = A[:, t] .* s_{t-1} + B[:, t],   t = 1, ..., T

with initial condition `s_0 = s0`.

Inputs:
- `A :: D×T`
- `B :: D×T`
- `s0 :: length-D`
- `S :: D×T` output buffer

This is the ground-truth implementation to compare against the parallel scan path.
It is intentionally simple and should be used in tests.
"""
function solve_affine_seq!(
    S::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector
)
    D, T = size(A)
    size(B) == (D, T) || throw(DimensionMismatch("B must have the same size as A"))
    size(S) == (D, T) || throw(DimensionMismatch("S must have size (D, T)"))
    length(s0) == D || throw(DimensionMismatch("length(s0) must equal size(A,1)"))

    T == 0 && return S

    @views begin
        S[:, 1] .= A[:, 1] .* s0 .+ B[:, 1]
        for t in 2:T
            S[:, t] .= A[:, t] .* S[:, t - 1] .+ B[:, t]
        end
    end
    return S
end

function solve_affine_seq(A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector)
    S = similar(A, promote_type(eltype(A), eltype(B), eltype(s0)), size(A))
    return solve_affine_seq!(S, A, B, s0)
end

function _check_scan_shapes(S, A, B, s0, ws::AffineScanWorkspace)
    D, T = size(A)
    size(B) == (D, T) || throw(DimensionMismatch("B must have the same size as A"))
    size(S) == (D, T) || throw(DimensionMismatch("S must have size (D, T)"))
    length(s0) == D || throw(DimensionMismatch("length(s0) must equal size(A,1)"))
    for buf in (ws.alpha, ws.beta, ws.alpha_new, ws.beta_new)
        size(buf) == (D, T) || throw(DimensionMismatch("workspace has wrong matrix size"))
    end
    Base.mightalias(S, A) && throw(ArgumentError("S must not alias A"))
    Base.mightalias(S, B) && throw(ArgumentError("S must not alias B"))
    return nothing
end

"""
    solve_affine_scan_diag!(S, A, B, s0, ws; check_shapes=true)

Parallel-doubling solver for the diagonal affine recurrence

    s_t = A[:, t] .* s_{t-1} + B[:, t],   t = 1, ..., T

This implements an associative scan over affine maps

    x ↦ a .* x + b

using the composition rule

    (a2, b2) ∘ (a1, b1) = (a2 .* a1, a2 .* b1 .+ b2)

The implementation is generic over `AbstractMatrix` and is intended to work on
CPU arrays and GPU arrays alike, as long as broadcasted assignment and
slicing/views are supported by the array type.

Notes:
- GPU arrays run a chunked scan in three KernelAbstractions kernels (see
  `_scan_chunked!`). Other arrays, including Reactant's, run O(log T) sweep
  levels of whole-array broadcasts, which XLA fuses under Reactant.
- `S` may alias neither `A` nor `B`.
- `ws` allows the scan to run without warm-path allocations.
"""
function solve_affine_scan_diag!(
    S::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    s0::AbstractVector,
    ws::AffineScanWorkspace;
    check_shapes::Bool=true,
)
    backend = _kernel_backend(S, A, B, s0)
    backend === nothing || return _scan_chunked!(backend, S, A, B, s0, ws)
    return _scan_doubling!(S, A, B, s0, ws; check_shapes=check_shapes)
end

#= Chunked scan of s_t = A[:, t] .* s_{t-1} .+ B[:, t] in three launches. Work-item
(d, k) composes the maps of chunk k for parameter d, one work-item per d carries the
state across chunks, and work-item (d, k) replays its chunk from that state. Each
state comes from the same sequential recursion as `solve_affine_seq!`. =#
@kernel function _chunk_summary_kernel!(Asum, Bsum, @Const(A), @Const(B), C::Int, T::Int)
    i = @index(Global, Linear)
    D = size(A, 1)
    d = (i - 1) % D + 1
    k = (i - 1) ÷ D + 1
    # (a, b) is the map s ↦ a * s + b of the steps so far; step t turns it into
    # s ↦ A[d, t] * (a * s + b) + B[d, t].
    a = one(eltype(Asum))
    b = zero(eltype(Bsum))
    @inbounds for t in ((k - 1) * C + 1):min(k * C, T)
        at = A[d, t]
        a = at * a
        b = at * b + B[d, t]
    end
    @inbounds Asum[d, k] = a
    @inbounds Bsum[d, k] = b
end

@kernel function _chunk_carry_kernel!(Sin, @Const(Asum), @Const(Bsum), @Const(s0))
    d = @index(Global, Linear)
    # Sin[d, k] is the state entering chunk k.
    @inbounds s = s0[d]
    @inbounds for k in axes(Sin, 2)
        Sin[d, k] = s
        s = Asum[d, k] * s + Bsum[d, k]
    end
end

@kernel function _chunk_replay_kernel!(S, @Const(A), @Const(B), @Const(Sin), C::Int, T::Int)
    i = @index(Global, Linear)
    D = size(A, 1)
    d = (i - 1) % D + 1
    k = (i - 1) ÷ D + 1
    @inbounds s = Sin[d, k]
    @inbounds for t in ((k - 1) * C + 1):min(k * C, T)
        s = A[d, t] * s + B[d, t]
        S[d, t] = s
    end
end

#= Kernels are built with a dynamic workgroup size and launched with this one. A
static size, `kernel!(backend, 256)`, does not infer, so each launch would dispatch
at run time, which costs more than a small scan. =#
const _WORKGROUP = 256

function _scan_chunked!(backend::KA.Backend, S, A, B, s0, ws::AffineScanWorkspace)
    D, Tn = size(A)
    #= Always check sizes, even with `check_shapes=false`: the kernels use
    `@inbounds`, so a mismatch would corrupt memory instead of throwing. =#
    _check_scan_shapes(S, A, B, s0, ws)
    (Tn == 0 || D == 0) && return S
    #= Each work-item of the carry kernel walks K = T / C steps, and each of the
    other two walks C steps; C ≈ √T keeps both short. =#
    C = max(32, isqrt(Tn))
    K = cld(Tn, C)
    Asum = view(ws.alpha, :, 1:K)
    Bsum = view(ws.beta, :, 1:K)
    Sin = view(ws.alpha_new, :, 1:K)
    wg = _WORKGROUP
    _chunk_summary_kernel!(backend)(
        Asum, Bsum, A, B, C, Tn; ndrange=D * K, workgroupsize=wg
    )
    _chunk_carry_kernel!(backend)(Sin, Asum, Bsum, s0; ndrange=D, workgroupsize=wg)
    _chunk_replay_kernel!(backend)(S, A, B, Sin, C, Tn; ndrange=D * K, workgroupsize=wg)
    return S
end

function _scan_doubling!(
    S::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    s0::AbstractVector,
    ws::AffineScanWorkspace;
    check_shapes::Bool=true,
)
    D, T = size(A)
    check_shapes && _check_scan_shapes(S, A, B, s0, ws)
    T == 0 && return S

    alpha = ws.alpha
    beta = ws.beta
    alpha_new = ws.alpha_new
    beta_new = ws.beta_new

    copyto!(alpha, A)
    copyto!(beta, B)

    offset = 1
    while offset < T
        last_level = (offset << 1) >= T
        @views begin
            if !last_level
                #=
                The destination already has the older prefix up to offset ÷ 2;
                only the newly exposed unchanged segment needs refreshing.
                =#
                copy_start = offset == 1 ? 1 : (offset >> 1) + 1
                alpha_new[:, copy_start:offset] .= alpha[:, copy_start:offset]
                beta_new[:, copy_start:offset] .= beta[:, copy_start:offset]
            end

            alpha_new[:, (offset + 1):T] .=
                alpha[:, (offset + 1):T] .* alpha[:, 1:(T - offset)]
            beta_new[:, (offset + 1):T] .=
                alpha[:, (offset + 1):T] .* beta[:, 1:(T - offset)] .+
                beta[:, (offset + 1):T]

            if last_level
                S[:, 1:offset] .= alpha[:, 1:offset] .* s0 .+ beta[:, 1:offset]
                S[:, (offset + 1):T] .=
                    alpha_new[:, (offset + 1):T] .* s0 .+ beta_new[:, (offset + 1):T]
                return S
            end
        end

        alpha, alpha_new = alpha_new, alpha
        beta, beta_new = beta_new, beta
        offset <<= 1
    end

    S .= alpha .* s0 .+ beta
    return S
end

function solve_affine_scan_diag!(
    S::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    s0::AbstractVector;
    check_shapes::Bool=true,
)
    ws = AffineScanWorkspace(A)
    return solve_affine_scan_diag!(S, A, B, s0, ws; check_shapes=check_shapes)
end

function solve_affine_scan_diag(A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector)
    T = promote_type(eltype(A), eltype(B), eltype(s0))
    S = similar(A, T, size(A))
    ws = AffineScanWorkspace(A)
    return solve_affine_scan_diag!(S, A, B, s0, ws)
end

"""
    affine_scan_residual(S, A, B, s0)

Return the maximum absolute residual of the recurrence

    S[:,1]      ?= A[:,1] .* s0       .+ B[:,1]
    S[:,t]      ?= A[:,t] .* S[:,t-1] .+ B[:,t],   t ≥ 2

Useful for debugging scan correctness independent of DEER.
"""
function affine_scan_residual(
    S::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector
)
    D, T = size(A)
    size(B) == (D, T) || throw(DimensionMismatch("B must have the same size as A"))
    size(S) == (D, T) || throw(DimensionMismatch("S must have size (D, T)"))
    length(s0) == D || throw(DimensionMismatch("length(s0) must equal size(A,1)"))

    T == 0 && return zero(promote_type(eltype(S), eltype(A), eltype(B), eltype(s0)))

    r = maximum(abs.(S[:, 1] .- (A[:, 1] .* s0 .+ B[:, 1])))
    @views for t in 2:T
        rt = maximum(abs.(S[:, t] .- (A[:, t] .* S[:, t - 1] .+ B[:, t])))
        r = max(r, rt)
    end
    return r
end

"""
    check_affine_scan(A, B, s0; atol=1e-6, rtol=1e-6)

Compare the sequential reference solver against the parallel-doubling solver.
Returns a named tuple with:
- `ok`
- `max_abs_err`
- `max_rel_err`
- `residual_seq`
- `residual_scan`
- `S_seq`
- `S_scan`

This is intended as a lightweight validation helper when plugging the scan into DEER.
"""
function check_affine_scan(
    A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector; atol=1e-6, rtol=1e-6
)
    S_seq = solve_affine_seq(A, B, s0)
    S_scan = solve_affine_scan_diag(A, B, s0)

    diff = abs.(S_seq .- S_scan)
    scale = atol .+ rtol .* abs.(S_seq)
    max_abs_err = isempty(diff) ? zero(eltype(diff)) : maximum(diff)
    max_rel_err = isempty(diff) ? zero(eltype(diff)) : maximum(diff ./ scale)
    residual_seq = affine_scan_residual(S_seq, A, B, s0)
    residual_scan = affine_scan_residual(S_scan, A, B, s0)

    return (
        ok=all(diff .<= scale),
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        residual_seq=residual_seq,
        residual_scan=residual_scan,
        S_seq=S_seq,
        S_scan=S_scan,
    )
end

end # module
