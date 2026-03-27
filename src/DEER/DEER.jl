module DEER

using Base.Threads: @threads, threadid
using LinearAlgebra
using DifferentiationInterface
using ADTypes: ADTypes, AbstractADType
import Mooncake: Mooncake
using Random

const DI = DifferentiationInterface
const DEFAULT_BACKEND = DI.AutoMooncake(; config=nothing)

"""
Deterministic recursion driven by a pre-generated tape.

- `step_fwd(x, tape_t)`: exact forward transition (may include MH accept/reject).
- `step_lin(x, tape_t, c...)`: surrogate used only for Jacobians.
- `consts(x, tape_t) -> Tuple`: returns constants `c...` passed into `step_lin`.
- `const_example`: example tuple of constants, used in `prepare`.
- `backend`: AD backend (any `ADTypes.AbstractADType`); defaults to `AutoMooncake`.
  Pass `AutoEnzyme()` or another GPU-compatible backend for GPU execution.
"""
struct TapedRecursion{Ff,Fl,Fc,Tt,Ce,AD<:AbstractADType}
    step_fwd::Ff
    step_lin::Fl
    consts::Fc
    tape::Vector{Tt}
    const_example::Ce
    backend::AD
end

"Backward-compatible constructor: uses the same step for forward + Jacobian, no constants."
function TapedRecursion(step, tape::Vector)
    return TapedRecursion(step, step, (_x, _tt) -> (), tape, (), DEFAULT_BACKEND)
end

"Main constructor."
function TapedRecursion(
    step_fwd, step_lin, tape::Vector;
    consts=(_x, _tt) -> (), const_example=(), backend::AbstractADType=DEFAULT_BACKEND,
)
    return TapedRecursion(step_fwd, step_lin, consts, tape, const_example, backend)
end

# Stable callable for DI (Jacobian always taken w.r.t. step_lin)
struct StepWithTape{R}
    rec::R
end
(s::StepWithTape)(x, tape, cs...) = s.rec.step_lin(x, tape, cs...)

"Prepare Jacobian for the surrogate step (reusable across t as long as tape element type is stable)."
function prepare(rec::TapedRecursion, x0::AbstractVector)
    f  = StepWithTape(rec)
    cs = rec.const_example
    return DI.prepare_jacobian(
        f, rec.backend, x0, DI.Constant(rec.tape[1]), (DI.Constant(c) for c in cs)...
    )
end

"Prepare pushforward (JVP) for the surrogate step_lin."
function prepare_pushforward(rec::TapedRecursion, x0::AbstractVector)
    f   = StepWithTape(rec)
    cs  = rec.const_example
    tx0 = (zero(x0),)
    return DI.prepare_pushforward(
        f, rec.backend, x0, tx0, DI.Constant(rec.tape[1]), (DI.Constant(c) for c in cs)...
    )
end

"Full Jacobian of surrogate step_lin(x, tape_t, consts...) w.r.t. x."
function jac_full(rec::TapedRecursion, prep, x::AbstractVector, t::Int)
    f  = StepWithTape(rec)
    cs = rec.consts(x, rec.tape[t])
    return DI.jacobian(
        f, prep, rec.backend, x, DI.Constant(rec.tape[t]), (DI.Constant(c) for c in cs)...
    )
end

"Diagonal of the surrogate Jacobian via full Jacobian computation."
function jac_diag(rec::TapedRecursion, prep, x::AbstractVector, t::Int)
    return diag(jac_full(rec, prep, x, t))
end

@inline function _rademacher!(z::AbstractVector{T}, rng::AbstractRNG) where {T}
    D    = length(z)
    bits = rand(rng, Bool, D)
    vals = Vector{T}(undef, D)
    for i in 1:D
        vals[i] = bits[i] ? one(T) : -one(T)
    end
    copyto!(z, vals)   # host-to-device when z is a CuVector; plain copy otherwise
    return z
end

function _jvp_step_lin(
    rec::TapedRecursion, prep_pf, x::AbstractVector, t::Int, v::AbstractVector
)
    f  = StepWithTape(rec)
    cs = rec.consts(x, rec.tape[t])
    tx = (v,)
    res = DI.pushforward(
        f, prep_pf, rec.backend, x, tx,
        DI.Constant(rec.tape[t]), (DI.Constant(c) for c in cs)...,
    )
    return res isa Tuple ? res[end] : res
end

"""
Stochastic (Hutchinson/Rademacher) estimator of diag(J), where J is the Jacobian of step_lin.

diag(J) ≈ (1/K) * Σ_k  z^(k) ⊙ (J z^(k)),   z_i ∈ {±1}

Keywords:
- probes: number of random probe vectors K (typical 1–4)
- rng: RNG for probes
- zbuf: optional preallocated buffer (length D) to avoid allocations
"""
function jac_diag_stoch(
    rec::TapedRecursion,
    prep_pf,
    x::AbstractVector,
    t::Int;
    probes::Int=1,
    rng::AbstractRNG=Random.default_rng(),
    zbuf::Union{Nothing,AbstractVector}=nothing,
)
    D      = length(x)
    FT     = float(eltype(x))
    probes ≥ 1 || throw(ArgumentError("probes must be ≥ 1"))

    z = zbuf === nothing ? Vector{FT}(undef, D) : zbuf
    length(z) == D || throw(ArgumentError("zbuf must have length D"))

    # d matches the array type of z (CuVector on GPU, Vector on CPU).
    d = fill!(similar(z, D), zero(FT))
    for _ in 1:probes
        _rademacher!(z, rng)
        jv = _jvp_step_lin(rec, prep_pf, x, t, z)
        @. d += z * jv
    end
    d .*= one(FT) / probes
    return d
end

"""
    solve_affine_scan_diag(A, B, s0)

Solve the diagonal affine recurrence `s_t = A[:,t] .* s_{t-1} + B[:,t]` for t=1..T
via an associative inclusive parallel-prefix (Hillis-Steele) scan.

**Complexity**: O(log T) sequential sweep levels; each level is a single broadcast
over all T columns — no loops or `@threads`.  The implementation is
**array-type-agnostic**: it works identically on CPU `Matrix`, GPU `CuMatrix`, or
any other `AbstractMatrix`.

Inputs:
- `A` :: D×T — per-step multiplicative coefficients
- `B` :: D×T — per-step additive offsets
- `s0` :: length-D initial state

Returns:
- `S` :: D×T (same type as A) where `S[:,t] = s_t`
"""
function solve_affine_scan_diag(A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector)
    D, T = size(A)
    size(B) == (D, T) || throw(ArgumentError("B must have the same size as A"))
    length(s0) == D   || throw(ArgumentError("s0 length must match size(A,1)"))

    # Work on copies; swap buffers each level to avoid allocating inside the loop.
    α    = copy(A)
    β    = copy(B)
    αnew = similar(α)
    βnew = similar(β)

    offset = 1
    while offset < T
        # Columns 1:offset are unchanged at this level.
        αnew[:, 1:offset]       .= α[:, 1:offset]
        βnew[:, 1:offset]       .= β[:, 1:offset]

        # Columns offset+1:T combine with their left neighbour at distance `offset`.
        #   αnew[:,t] = α[:,t] * α[:,t-offset]
        #   βnew[:,t] = α[:,t] * β[:,t-offset] + β[:,t]
        αnew[:, offset+1:T] .= α[:, offset+1:T] .* α[:, 1:T-offset]
        βnew[:, offset+1:T] .= α[:, offset+1:T] .* β[:, 1:T-offset] .+ β[:, offset+1:T]

        α, αnew = αnew, α
        β, βnew = βnew, β
        offset <<= 1
    end

    # Apply initial condition: S[:,t] = α[:,t] * s0 + β[:,t]
    S = similar(A, D, T)
    S .= α .* reshape(s0, D, 1) .+ β
    return S
end

"""
Compute one DEER update given a trajectory guess `S` (D×T).
Returns `S_new` (D×T).

The per-timestep Jacobian computation uses the AD backend stored in `rec.backend`.
For GPU execution, construct the `TapedRecursion` with a GPU-compatible backend
(e.g. `AutoEnzyme()`).

Keywords:
- `jacobian`: `:diag`, `:stoch_diag`, or `:full`
- `damping`: in (0,1]
"""
function deer_update(
    rec::TapedRecursion,
    s0_in::AbstractVector,
    S::AbstractMatrix,
    prep;
    jacobian::Symbol=:diag,
    damping::Real=1.0,
    probes::Int=1,
    rng::AbstractRNG=Random.default_rng(),
)
    s0   = vec(s0_in)
    D, T = size(S)
    length(s0) == D        || throw(ArgumentError("S must be D×T with D=length(s0)"))
    T == length(rec.tape)  || throw(ArgumentError("size(S,2) must match length(rec.tape)"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))
    probes ≥ 1             || throw(ArgumentError("probes must be ≥ 1"))

    FT = float(eltype(s0))

    if jacobian === :diag || jacobian === :stoch_diag
        # A and B match the array type of S (CuMatrix on GPU, Matrix on CPU).
        A = similar(S)
        B = similar(S)

        nt   = Base.Threads.maxthreadid()
        # zbufs for stoch_diag: match array type of s0 so GPU backends work correctly.
        zbufs = jacobian === :stoch_diag ? [similar(s0, D) for _ in 1:nt] : nothing
        rngs  = if jacobian === :stoch_diag
            seeds = rand(rng, UInt, nt)
            [MersenneTwister(seeds[i]) for i in 1:nt]
        else
            nothing
        end

        @threads for t in 1:T
            tid  = threadid()
            # S[:, t-1] returns a concrete column copy — Vector on CPU, CuVector on GPU.
            xbar = t == 1 ? s0 : S[:, t - 1]

            jt = if jacobian === :diag
                jac_diag(rec, prep, xbar, t)
            else
                jac_diag_stoch(
                    rec, prep, xbar, t; probes=probes, rng=rngs[tid], zbuf=zbufs[tid],
                )
            end

            ft = rec.step_fwd(xbar, rec.tape[t])
            # Broadcast column assignment: works on both CPU Matrix and GPU CuMatrix.
            view(A, :, t) .= jt
            view(B, :, t) .= ft .- jt .* xbar
        end

        S_new = solve_affine_scan_diag(A, B, s0)
        if damping != 1.0
            @. S_new = (1 - damping) * S + damping * S_new
        end
        return S_new

    elseif jacobian === :full
        A_mats = Vector{Matrix{FT}}(undef, T)
        b      = Matrix{FT}(undef, D, T)
        xbuf   = zeros(FT, D)

        for t in 1:T
            xbar = if t == 1
                s0
            else
                for i in 1:D; xbuf[i] = S[i, t - 1]; end
                xbuf
            end

            Jt      = jac_full(rec, prep, xbar, t)
            A_mats[t] = Jt
            ft      = rec.step_fwd(xbar, rec.tape[t])
            tmp     = Jt * xbar
            for i in 1:D; b[i, t] = ft[i] - tmp[i]; end
        end

        S_new  = Matrix{FT}(undef, D, T)
        s_prev = copy(s0)
        for t in 1:T
            s_prev      = A_mats[t] * s_prev .+ view(b, :, t)
            S_new[:, t] .= s_prev
        end
        if damping != 1.0
            S_new .= (1 - damping) .* S .+ damping .* S_new
        end
        return S_new
    else
        throw(ArgumentError("jacobian must be :diag, :stoch_diag, or :full"))
    end
end

@inline function _maxabs(x)
    m = zero(real(eltype(x)))
    for i in eachindex(x)
        v = abs(x[i])
        m = ifelse(v > m, v, m)
    end
    return m
end

@inline function _maxabsdiff(x, y)
    m = zero(real(promote_type(eltype(x), eltype(y))))
    for i in eachindex(x, y)
        v = abs(x[i] - y[i])
        m = ifelse(v > m, v, m)
    end
    return m
end

"""
Run DEER iterations until convergence.

Returns the solved trajectory `S :: D×T` (same array type as the initial state,
so passing a `CuVector` for `s0_in` will yield a `CuMatrix`).

# GPU use
1. Construct a `DensityModel` whose `logdensity` and `grad_logdensity` operate on
   GPU arrays.
2. Pass `backend = AutoEnzyme()` (or another GPU-compatible `ADTypes` backend) to
   `DEERSampler` so that DEER uses it when building the `TapedRecursion`.
3. Pass a GPU vector as `initial_params` (or use a `DEERSampler` with a float type
   matching the GPU array element type).

The `solve_affine_scan_diag` kernel runs as pure broadcasts and requires no
backend-specific GPU code; only the Jacobian computation via `DI` depends on the
chosen backend.

Keywords:
- `init`: initial trajectory guess D×T (default: repeat s0 across columns)
- `tol_abs`, `tol_rel`: stopping tolerances (∞-norm per time step)
- `maxiter`, `jacobian`, `damping`, `probes`, `return_info`
"""
function solve(
    rec::TapedRecursion,
    s0_in::AbstractVector;
    init=nothing,
    tol_abs::Real=1e-4,
    tol_rel::Real=1e-3,
    maxiter::Int=200,
    jacobian::Symbol=:diag,
    damping::Real=1.0,
    probes::Int=1,
    rng::AbstractRNG=Random.default_rng(),
    return_info::Bool=false,
)
    s0 = vec(s0_in)
    D  = length(s0)
    T  = length(rec.tape)

    maxiter ≥ 1 || throw(ArgumentError("maxiter must be ≥ 1"))
    tol_abs ≥ 0 || throw(ArgumentError("tol_abs must be ≥ 0"))
    tol_rel ≥ 0 || throw(ArgumentError("tol_rel must be ≥ 0"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))

    # Initial trajectory guess: preserve array type of s0
    S = if init === nothing
        S0 = similar(s0, D, T)
        S0 .= reshape(s0, D, 1)
        S0
    else
        size(init) == (D, T) || throw(ArgumentError("init must be size (D,T)"))
        copy(init)
    end

    prep = if jacobian === :stoch_diag
        prepare_pushforward(rec, s0)
    elseif jacobian === :full || jacobian === :diag
        prepare(rec, s0)
    else
        nothing
    end

    converged   = false
    last_metric = Inf
    iters       = 0

    for iter in 1:maxiter
        iters = iter
        S_new = deer_update(
            rec, s0, S, prep; jacobian=jacobian, damping=damping, probes=probes, rng=rng,
        )

        # Single pair of reductions over the full D×T matrix — one GPU kernel each.
        # This replaces T per-column scalar loops and is efficient on both CPU and GPU.
        Δ_max   = maximum(abs.(S_new .- S))
        S_scale = tol_abs + tol_rel * maximum(abs.(S_new))
        metric  = Δ_max / S_scale

        S           = S_new
        last_metric = metric
        metric ≤ 1 && (converged = true; break)
    end

    return_info || return S
    return S, (
        converged = converged,
        iters     = iters,
        metric    = last_metric,
        jacobian  = jacobian,
        damping   = damping,
        probes    = probes,
    )
end

end # module DEER
