module DEER

using Base.Threads: @threads, threadid
using LinearAlgebra
using DifferentiationInterface
import Mooncake: Mooncake
using Random

const DI = DifferentiationInterface
backend = DI.AutoMooncake(; config=nothing)

"""
Deterministic recursion driven by a pre-generated tape.

- `step_fwd(x, tape_t)`: exact forward transition (may include MH accept/reject).
- `step_lin(x, tape_t, c...)`: surrogate used only for Jacobians.
- `consts(x, tape_t) -> Tuple`: returns constants `c...` passed into `step_lin` as DI.Constant.
- `const_example`: example tuple of constants, used in `prepare`.
"""
struct TapedRecursion{Ff,Fl,Fc,Tt,Ce}
    step_fwd::Ff
    step_lin::Fl
    consts::Fc
    tape::Vector{Tt}
    const_example::Ce
end

"Backward-compatible constructor: uses the same step for forward + Jacobian, and no extra constants."
TapedRecursion(step, tape::Vector) = TapedRecursion(step, step, (_x, _tt)->(), tape, ())

"Main constructor."
function TapedRecursion(
    step_fwd, step_lin, tape::Vector; consts=(_x, _tt)->(), const_example=()
)
    return TapedRecursion(step_fwd, step_lin, consts, tape, const_example)
end

# Stable callable for DI (Jacobian always taken w.r.t. step_lin)
struct StepWithTape{R}
    rec::R
end
(s::StepWithTape)(x, tape, cs...) = s.rec.step_lin(x, tape, cs...)

"Prepare Jacobian for the surrogate step (reusable across t as long as tape element type is stable)."
function prepare(rec::TapedRecursion, x0::AbstractVector)
    f = StepWithTape(rec)
    cs = rec.const_example
    return DI.prepare_jacobian(
        f, backend, x0, DI.Constant(rec.tape[1]), (DI.Constant(c) for c in cs)...
    )
end

"Prepare pushforward (JVP) for the surrogate step_lin."
function prepare_pushforward(rec::TapedRecursion, x0::AbstractVector)
    f = StepWithTape(rec)
    cs = rec.const_example
    # tx is a tuple of tangents; we supply an example tangent
    tx0 = (zero(x0),)
    return DI.prepare_pushforward(
        f, backend, x0, tx0, DI.Constant(rec.tape[1]), (DI.Constant(c) for c in cs)...
    )
end

"Full Jacobian of surrogate step_lin(x, tape_t, consts...) w.r.t. x."
function jac_full(rec::TapedRecursion, prep, x::AbstractVector, t::Int)
    f = StepWithTape(rec)
    cs = rec.consts(x, rec.tape[t])
    return DI.jacobian(
        f, prep, backend, x, DI.Constant(rec.tape[t]), (DI.Constant(c) for c in cs)...
    )
end

"Diagonal of the surrogate Jacobian (debug/correctness mode; not scalable if it computes full J)."
function jac_diag(rec::TapedRecursion, prep, x::AbstractVector, t::Int)
    return diag(jac_full(rec, prep, x, t))
end

@inline function _rademacher!(z::AbstractVector, rng::AbstractRNG)
    for i in eachindex(z)
        z[i] = rand(rng, Bool) ? 1.0 : -1.0
    end
    return z
end

function _jvp_step_lin(
    rec::TapedRecursion, prep_pf, x::AbstractVector, t::Int, v::AbstractVector
)
    f = StepWithTape(rec)
    cs = rec.consts(x, rec.tape[t])

    # pushforward expects tx as a tuple of tangents
    tx = (v,)

    res = DI.pushforward(
        f,
        prep_pf,
        backend,
        x,
        tx,
        DI.Constant(rec.tape[t]),
        (DI.Constant(c) for c in cs)...,
    )

    return res isa Tuple ? res[end] : res
end

"""
Stochastic (Hutchinson/Rademacher) estimator of diag(J), where J is the Jacobian of step_lin.

diag(J) ≈ (1/K) * Σ_k  z^(k) ⊙ (J z^(k)),   z_i ∈ {±1}

Keywords:
- probes: number of random probe vectors K (typical 1–4)
- rng: RNG for probes
- zbuf, jbuf: optional preallocated buffers (length D) to avoid allocations
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
    D = length(x)
    probes ≥ 1 || throw(ArgumentError("probes must be ≥ 1"))

    z = zbuf === nothing ? Vector{Float64}(undef, D) : zbuf
    length(z) == D || throw(ArgumentError("zbuf must have length D"))

    d = zeros(Float64, D)

    for k in 1:probes
        _rademacher!(z, rng)
        jv = _jvp_step_lin(rec, prep_pf, x, t, z)   # J*z
        for i in 1:D
            d[i] += z[i] * jv[i]
        end
    end

    invK = 1.0 / probes
    for i in 1:D
        d[i] *= invK
    end
    return d
end

"""
Solve s_t = a_t .* s_{t-1} + b_t for t=1..T using an associative inclusive scan.

Inputs:
- A :: D×T matrix, A[:,t] = a_t
- B :: D×T matrix, B[:,t] = b_t
- s0 :: length-D vector

Returns:
- S :: D×T matrix, S[:,t] = s_t
"""
function solve_affine_scan_diag(A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector)
    D, T = size(A)
    size(B) == (D, T) || throw(ArgumentError("B must have the same size as A"))
    length(s0) == D || throw(ArgumentError("s0 length must match size(A,1)"))

    α = Matrix{Float64}(A)         # working prefixes
    β = Matrix{Float64}(B)
    αnew = similar(α)
    βnew = similar(β)

    offset = 1
    while offset < T
        @threads for t in 1:T
            if t > offset
                for i in 1:D
                    ai = α[i, t]
                    αnew[i, t] = ai * α[i, t - offset]
                    βnew[i, t] = ai * β[i, t - offset] + β[i, t]
                end
            else
                for i in 1:D
                    αnew[i, t] = α[i, t]
                    βnew[i, t] = β[i, t]
                end
            end
        end
        α, αnew = αnew, α
        β, βnew = βnew, β
        offset <<= 1
    end

    S = Matrix{Float64}(undef, D, T)
    @threads for t in 1:T
        for i in 1:D
            S[i, t] = α[i, t] * s0[i] + β[i, t]
        end
    end
    return S
end

"""
Compute one DEER update given a trajectory guess S (D×T).
Returns S_new (D×T).

Keywords:
- jacobian: :diag or :full
- damping: in (0,1]
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
    s0 = vec(s0_in)
    D, T = size(S)
    length(s0) == D || throw(ArgumentError("S must be D×T with D=length(s0)"))
    T == length(rec.tape) || throw(ArgumentError("size(S,2) must match length(rec.tape)"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))
    probes ≥ 1 || throw(ArgumentError("probes must be ≥ 1"))

    if jacobian === :diag || jacobian === :stoch_diag
        A = Matrix{Float64}(undef, D, T)
        B = Matrix{Float64}(undef, D, T)

        nt = Base.Threads.maxthreadid()

        # scratch vector per thread for xbar
        xbufs = [zeros(Float64, D) for _ in 1:nt]

        # scratch per thread for stochastic diag probes
        zbufs = jacobian === :stoch_diag ? [zeros(Float64, D) for _ in 1:nt] : nothing

        # per-thread RNGs for reproducibility + thread safety
        rngs = if jacobian === :stoch_diag
            # Seed thread RNGs from the provided rng on the main thread (deterministic given rng).
            seeds = rand(rng, UInt, nt)
            [MersenneTwister(seeds[i]) for i in 1:nt]
        else
            nothing
        end

        @threads for t in 1:T
            tid = threadid()

            xbar = if t == 1
                s0
            else
                xb = xbufs[tid]
                for i in 1:D
                    xb[i] = S[i, t - 1]
                end
                xb
            end

            jt = if jacobian === :diag
                jac_diag(rec, prep, xbar, t)
            else
                jac_diag_stoch(
                    rec, prep, xbar, t; probes=probes, rng=rngs[tid], zbuf=zbufs[tid]
                )
            end

            ft = rec.step_fwd(xbar, rec.tape[t])

            for i in 1:D
                A[i, t] = jt[i]
                B[i, t] = ft[i] - jt[i] * xbar[i]
            end
        end

        S_new = solve_affine_scan_diag(A, B, s0)

        if damping != 1.0
            @threads for t in 1:T
                for i in 1:D
                    S_new[i, t] = (1 - damping) * S[i, t] + damping * S_new[i, t]
                end
            end
        end

        return S_new

    elseif jacobian === :full
        A = Vector{Matrix{Float64}}(undef, T)
        b = Matrix{Float64}(undef, D, T)

        xbuf = zeros(Float64, D)  # sequential scratch

        for t in 1:T
            xbar = if t == 1
                s0
            else
                for i in 1:D
                    xbuf[i] = S[i, t - 1]
                end
                xbuf
            end

            Jt = jac_full(rec, prep, xbar, t)
            A[t] = Jt

            ft = rec.step_fwd(xbar, rec.tape[t])
            tmp = Jt * xbar
            for i in 1:D
                b[i, t] = ft[i] - tmp[i]
            end
        end

        S_new = Matrix{Float64}(undef, D, T)
        s_prev = copy(s0)
        for t in 1:T
            s_prev = A[t] * s_prev .+ view(b, :, t)
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
    m = 0.0
    for i in eachindex(x)
        v = abs(x[i])
        m = ifelse(v > m, v, m)
    end
    return m
end

@inline function _maxabsdiff(x, y)
    m = 0.0
    for i in eachindex(x, y)
        v = abs(x[i] - y[i])
        m = ifelse(v > m, v, m)
    end
    return m
end

"""
Run DEER iterations until convergence.

Returns:
- S :: D×T matrix trajectory (S[:,t] is state at time t)

Keywords:
- init: initial trajectory guess, D×T (default: repeat s0)
- tol_abs, tol_rel: stopping tolerances (∞-norm per time step)
- maxiter
- jacobian: :diag or :full
- damping
- return_info
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
    D = length(s0)
    T = length(rec.tape)

    maxiter ≥ 1 || throw(ArgumentError("maxiter must be ≥ 1"))
    tol_abs ≥ 0 || throw(ArgumentError("tol_abs must be ≥ 0"))
    tol_rel ≥ 0 || throw(ArgumentError("tol_rel must be ≥ 0"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))

    S = if init === nothing
        # repeat s0 across time
        M = Matrix{Float64}(undef, D, T)
        @threads for t in 1:T
            M[:, t] .= s0
        end
        M
    else
        size(init) == (D, T) || throw(ArgumentError("init must be size (D,T)"))
        Matrix{Float64}(init)
    end

    prep = if jacobian === :stoch_diag
        prepare_pushforward(rec, s0)
    else
        prepare(rec, s0)
    end

    converged = false
    last_metric = Inf
    iters = 0

    for iter in 1:maxiter
        iters = iter
        S_new = deer_update(
            rec, s0, S, prep; jacobian=jacobian, damping=damping, probes=probes, rng=rng
        )

        metric = 0.0
        for t in 1:T
            xnew = view(S_new, :, t)
            xold = view(S, :, t)
            Δ = _maxabsdiff(xnew, xold)
            scale = tol_abs + tol_rel * _maxabs(xnew)
            metric = max(metric, Δ / scale)
        end

        S = S_new
        last_metric = metric

        if metric ≤ 1.0
            converged = true
            break
        end
    end

    if return_info
        return S,
        (
            converged=converged,
            iters=iters,
            metric=last_metric,
            jacobian=jacobian,
            damping=damping,
            probes=probes,
        )
    else
        return S
    end
end

end # module DEER
