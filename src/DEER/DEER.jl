module DEER

using LinearAlgebra
using DifferentiationInterface
using ADTypes: ADTypes, AbstractADType
import Enzyme: Enzyme
using Random

const DI = DifferentiationInterface
# Enzyme forward-mode is optimal for JVPs used in :stoch_diag.
# NOTE: On Windows, Enzyme CPU has a known LLVM unwinding issue (pinned v0.13.131).
# For Windows CPU use, construct samplers with backend=AutoForwardDiff() explicitly.
const DEFAULT_BACKEND = ADTypes.AutoEnzyme(; mode=Enzyme.Forward)

"""
Deterministic recursion driven by a pre-generated tape.

- `step_fwd(x, tape_t)`: exact forward transition (may include MH accept/reject).
- `step_lin(x, tape_t, c...)`: surrogate used only for Jacobians (via AD).
- `jvp`: optional explicit JVP `(x, tape_t, v) -> J·v`.  When provided, bypasses AD
  entirely for `:stoch_diag` and `:diag` modes — useful when `step_lin` contains
  nested AD (e.g. gradlogp computed by reverse-mode AD).  Defaults to `nothing`.
- `consts(x, tape_t) -> Tuple`: returns constants `c...` passed into `step_lin`.
- `const_example`: example tuple of constants, used in `prepare`.
- `backend`: AD backend (any `ADTypes.AbstractADType`); defaults to `AutoEnzyme(Forward)`.
  Pass a GPU-compatible backend for GPU execution.
"""
struct TapedRecursion{Ff,Fl,Fj,Fc,Tt,Ce,AD<:AbstractADType}
    step_fwd::Ff
    step_lin::Fl
    jvp::Fj          # explicit (x, te, v) -> J·v, or Nothing
    consts::Fc
    tape::Vector{Tt}
    const_example::Ce
    backend::AD
end

"Backward-compatible constructor: uses the same step for forward + Jacobian, no constants."
function TapedRecursion(step, tape::Vector)
    return TapedRecursion(step, step, nothing, (_x, _tt) -> (), tape, (), DEFAULT_BACKEND)
end

"Main constructor."
function TapedRecursion(
    step_fwd,
    step_lin,
    tape::Vector;
    jvp=nothing,
    consts=(_x, _tt) -> (),
    const_example=(),
    backend::AbstractADType=DEFAULT_BACKEND,
)
    return TapedRecursion(step_fwd, step_lin, jvp, consts, tape, const_example, backend)
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
        f, rec.backend, x0, DI.Constant(rec.tape[1]), (DI.Constant(c) for c in cs)...
    )
end

"Prepare pushforward (JVP) for the surrogate step_lin."
function prepare_pushforward(rec::TapedRecursion, x0::AbstractVector)
    f = StepWithTape(rec)
    cs = rec.const_example
    tx0 = (zero(x0),)
    return DI.prepare_pushforward(
        f, rec.backend, x0, tx0, DI.Constant(rec.tape[1]), (DI.Constant(c) for c in cs)...
    )
end

"Full Jacobian of surrogate step_lin(x, tape_t, consts...) w.r.t. x."
function jac_full(rec::TapedRecursion, prep, x::AbstractVector, t::Int)
    f = StepWithTape(rec)
    cs = rec.consts(x, rec.tape[t])
    return DI.jacobian(
        f, prep, rec.backend, x, DI.Constant(rec.tape[t]), (DI.Constant(c) for c in cs)...
    )
end

"Diagonal of the surrogate Jacobian via full Jacobian computation."
function jac_diag(rec::TapedRecursion, prep, x::AbstractVector, t::Int)
    return diag(jac_full(rec, prep, x, t))
end

"""
Diagonal of the surrogate Jacobian via D JVP calls with standard basis vectors.

GPU-compatible alternative to `jac_diag`: ForwardDiff's `jacobian` uses scalar
indexing to seed Dual numbers (incompatible with CuArrays), but `pushforward`
constructs Duals via element-wise broadcast and works on GPU.

For standard basis e_i:  diag(J)[i] = (J e_i)[i],  accumulated as  d .+= e_i .* (J e_i).
"""
function jac_diag_via_jvps(rec::TapedRecursion, prep_pf, x::AbstractVector, t::Int)
    D = length(x)
    FT = eltype(x)
    # Copy the D×D identity matrix to the same device as x in one transfer.
    I_buf = similar(x, D, D)
    copyto!(I_buf, Matrix{FT}(I, D, D))
    d = similar(x)
    fill!(d, zero(FT))
    for i in 1:D
        ei = view(I_buf, :, i)        # column slice — no scalar indexing
        jv = _jvp_step_lin(rec, prep_pf, x, t, ei)
        d .+= ei .* jv                # accumulate exact J[i,i] at position i
    end
    return d
end

@inline function _rademacher!(z::AbstractVector{T}, rng::AbstractRNG) where {T}
    D = length(z)
    bits = rand(rng, Bool, D)
    vals = Vector{T}(undef, D)
    for i in 1:D
        vals[i] = bits[i] ? one(T) : -one(T)
    end
    copyto!(z, vals)   # host-to-device when z is a CuVector; plain copy otherwise
    return z
end

"""
Compute a single JVP of `f` at `x` in direction `v` using `backend`, preparing
internally on every call.  Suitable for use inside explicit JVP closures where
the call site (e.g. `mala_step_surrogate_sigmoid_jvp`) manages its own loop and
re-preparation overhead is acceptable.
"""
function _hvp_nopre(
    f, backend::AbstractADType, x::AbstractVector, v::AbstractVector
)
    prep = DI.prepare_pushforward(f, backend, x, (v,))
    res = DI.pushforward(f, prep, backend, x, (v,))
    return res isa Tuple ? first(res) : res
end

function _jvp_step_lin(
    rec::TapedRecursion, prep_pf, x::AbstractVector, t::Int, v::AbstractVector
)
    if rec.jvp !== nothing
        return rec.jvp(x, rec.tape[t], v)
    end
    f = StepWithTape(rec)
    cs = rec.consts(x, rec.tape[t])
    tx = (v,)
    res = DI.pushforward(
        f,
        prep_pf,
        rec.backend,
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
    D = length(x)
    FT = float(eltype(x))
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
    length(s0) == D || throw(ArgumentError("s0 length must match size(A,1)"))

    # Work on copies; swap buffers each level to avoid allocating inside the loop.
    α = copy(A)
    β = copy(B)
    αnew = similar(α)
    βnew = similar(β)

    offset = 1
    while offset < T
        # Columns 1:offset are unchanged at this level.
        αnew[:, 1:offset] .= α[:, 1:offset]
        βnew[:, 1:offset] .= β[:, 1:offset]

        # Columns offset+1:T combine with their left neighbour at distance `offset`.
        #   αnew[:,t] = α[:,t] * α[:,t-offset]
        #   βnew[:,t] = α[:,t] * β[:,t-offset] + β[:,t]
        αnew[:, (offset + 1):T] .= α[:, (offset + 1):T] .* α[:, 1:(T - offset)]
        βnew[:, (offset + 1):T] .=
            α[:, (offset + 1):T] .* β[:, 1:(T - offset)] .+ β[:, (offset + 1):T]

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
    s0 = vec(s0_in)
    D, T = size(S)
    length(s0) == D || throw(ArgumentError("S must be D×T with D=length(s0)"))
    T == length(rec.tape) || throw(ArgumentError("size(S,2) must match length(rec.tape)"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))
    probes ≥ 1 || throw(ArgumentError("probes must be ≥ 1"))

    FT = float(eltype(s0))

    if jacobian === :diag || jacobian === :stoch_diag
        # A and B match the array type of S (CuMatrix on GPU, Matrix on CPU).
        A = similar(S)
        B = similar(S)

        # GPU arrays (CuVector, etc.) use a sequential loop:
        #   (a) AD backends store mutable prep buffers; concurrent Julia tasks on GPU
        #       arrays would race against device memory.
        #   (b) GPU kernels are internally parallel; host-side task parallelism adds
        #       no benefit and can break device memory safety.
        on_gpu = !(s0 isa Array)

        if on_gpu
            zbuf_gpu = jacobian === :stoch_diag ? similar(s0, D) : nothing
            for t in 1:T
                xbar = t == 1 ? s0 : S[:, t - 1]
                jt = if jacobian === :diag
                    jac_diag_via_jvps(rec, prep, xbar, t)
                else
                    jac_diag_stoch(rec, prep, xbar, t; probes=probes, rng=rng, zbuf=zbuf_gpu)
                end
                ft = rec.step_fwd(xbar, rec.tape[t])
                view(A, :, t) .= jt
                view(B, :, t) .= ft .- jt .* xbar
            end
        else
            # Pre-generate one seed per time step before spawning tasks.
            # Sharing `rng` across concurrently running tasks would be a data race.
            # Using @threads + threadid() to index per-thread buffers is an unsafe
            # pattern in Julia (tasks can migrate across threads); @spawn with
            # task-local state is the correct approach.
            task_seeds = jacobian === :stoch_diag ? rand(rng, UInt, T) : nothing

            tasks = map(1:T) do t
                seed_t = jacobian === :stoch_diag ? task_seeds[t] : nothing
                Threads.@spawn begin
                    # Task-local state — each spawned task owns its own RNG + buffer.
                    zbuf_t = jacobian === :stoch_diag ? similar(s0, D) : nothing
                    rng_t = jacobian === :stoch_diag ? MersenneTwister(seed_t) : nothing
                    # S[:, t-1] returns a column copy — Vector on CPU.
                    xbar = t == 1 ? s0 : S[:, t - 1]
                    jt = if jacobian === :diag
                        # Explicit JVP: use JVP-based diagonal (avoids outer AD layer).
                        # Otherwise: full Jacobian via DI (chunked ForwardDiff on CPU).
                        if rec.jvp !== nothing
                            jac_diag_via_jvps(rec, prep, xbar, t)
                        else
                            jac_diag(rec, prep, xbar, t)
                        end
                    else
                        jac_diag_stoch(
                            rec,
                            prep,
                            xbar,
                            t;
                            probes=probes,
                            rng=rng_t,
                            zbuf=zbuf_t,
                        )
                    end
                    ft = rec.step_fwd(xbar, rec.tape[t])
                    (jt, ft .- jt .* xbar)
                end
            end
            for (t, task) in enumerate(tasks)
                jt, bt = fetch(task)
                view(A, :, t) .= jt
                view(B, :, t) .= bt
            end
        end

        S_new = solve_affine_scan_diag(A, B, s0)
        if damping != 1.0
            @. S_new = (1 - damping) * S + damping * S_new
        end
        return S_new

    elseif jacobian === :full
        A_mats = Vector{Matrix{FT}}(undef, T)
        b = Matrix{FT}(undef, D, T)
        xbuf = zeros(FT, D)

        for t in 1:T
            xbar = if t == 1
                s0
            else
                for i in 1:D
                    ;
                    xbuf[i] = S[i, t - 1];
                end
                xbuf
            end

            Jt = jac_full(rec, prep, xbar, t)
            A_mats[t] = Jt
            ft = rec.step_fwd(xbar, rec.tape[t])
            tmp = Jt * xbar
            for i in 1:D
                ;
                b[i, t] = ft[i] - tmp[i];
            end
        end

        S_new = Matrix{FT}(undef, D, T)
        s_prev = copy(s0)
        for t in 1:T
            s_prev = A_mats[t] * s_prev .+ view(b, :, t)
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
   `ParallelMALASampler` so that DEER uses it when building the `TapedRecursion`.
3. Pass a GPU vector as `initial_params` (or use a `ParallelMALASampler` with a float type
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
    D = length(s0)
    T = length(rec.tape)

    maxiter ≥ 1 || throw(ArgumentError("maxiter must be ≥ 1"))
    tol_abs ≥ 0 || throw(ArgumentError("tol_abs must be ≥ 0"))
    tol_rel ≥ 0 || throw(ArgumentError("tol_rel must be ≥ 0"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))

    # GPU force-override: on GPU arrays, :diag runs D sequential JVPs which is
    # prohibitively slow. Force :stoch_diag + probes=1 (1 JVP total), matching
    # the paper's GPU experiments (Section 5.2, stochastic quasi-DEER).
    on_gpu = !(s0 isa Array)
    if on_gpu && (jacobian !== :stoch_diag || probes != 1)
        @warn "ParallelMCMC: GPU detected — overriding to jacobian=:stoch_diag, probes=1 (paper recommendation)"
        jacobian = :stoch_diag
        probes = 1
    end

    # Initial trajectory guess: preserve array type of s0
    S = if init === nothing
        S0 = similar(s0, D, T)
        S0 .= reshape(s0, D, 1)
        S0
    else
        size(init) == (D, T) || throw(ArgumentError("init must be size (D,T)"))
        copy(init)
    end

    # When an explicit JVP is provided, DI prep is not needed for :diag/:stoch_diag.
    # :full still requires the full Jacobian prep (DI.jacobian path).
    # GPU arrays use JVP-based diagonal (broadcast-safe); CPU :diag uses full Jacobian.
    prep = if jacobian === :full
        prepare(rec, s0)
    elseif rec.jvp !== nothing
        nothing   # explicit JVP bypasses DI entirely
    elseif jacobian === :stoch_diag || (jacobian === :diag && !(s0 isa Array))
        prepare_pushforward(rec, s0)
    elseif jacobian === :diag
        prepare(rec, s0)
    else
        nothing
    end

    converged = false
    last_metric = Inf
    iters = 0

    for iter in 1:maxiter
        iters = iter
        S_new = deer_update(
            rec, s0, S, prep; jacobian=jacobian, damping=damping, probes=probes, rng=rng
        )

        # Single pair of reductions over the full D×T matrix — one GPU kernel each.
        # This replaces T per-column scalar loops and is efficient on both CPU and GPU.
        Δ_max = maximum(abs.(S_new .- S))
        S_scale = tol_abs + tol_rel * maximum(abs.(S_new))
        metric = Δ_max / S_scale

        S = S_new
        last_metric = metric
        metric ≤ 1 && (converged=true; break)
    end

    return_info || return S
    return S,
    (
        converged=converged,
        iters=iters,
        metric=last_metric,
        jacobian=jacobian,
        damping=damping,
        probes=probes,
    )
end

end # module DEER
