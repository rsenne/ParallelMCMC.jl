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
- `jvp(x, tape_t, v)`: explicit JVP `(x, tape_t, v) -> J·v` of the differentiable
  surrogate step.  Required; must not call AD through `step_fwd`'s control flow.
- `fwd_and_jvp(x, tape_t, v)`: optional fused function `-> (x_next, J·v)` that
  shares primal computation between the forward step and the JVP.  When provided,
  `deer_update` uses it for the first (or only) Hutchinson probe, halving the number
  of `gradlogp` and `logp` evaluations per time step.  Defaults to `nothing`.
"""
struct TapedRecursion{Ff,Fj,Ffj,Tt}
    step_fwd::Ff        # (x, tape_t) -> x_next
    jvp::Fj             # (x, tape_t, v) -> J·v  (explicit, required)
    fwd_and_jvp::Ffj    # (x, tape_t, v) -> (x_next, J·v), or Nothing
    tape::Vector{Tt}
end

"Main constructor."
function TapedRecursion(
    step_fwd,
    jvp,
    tape::Vector;
    fwd_and_jvp=nothing,
)
    return TapedRecursion(step_fwd, jvp, fwd_and_jvp, tape)
end

"""
Compute a single JVP of `f` at `x` in direction `v` using `backend`, preparing
internally on every call.  Used inside explicit JVP closures (e.g. inside
`mala_step_surrogate_sigmoid_jvp`) to compute HVPs of `gradlogp`.
"""
function _hvp_nopre(
    f, backend::AbstractADType, x::AbstractVector, v::AbstractVector
)
    prep = DI.prepare_pushforward(f, backend, x, (v,))
    res = DI.pushforward(f, prep, backend, x, (v,))
    return res isa Tuple ? first(res) : res
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
Diagonal of the Jacobian of `rec.jvp` w.r.t. `x` via D explicit JVP calls with
standard basis vectors.  Exact but O(D) JVP evaluations.

GPU-compatible: uses column slices of a D×D identity buffer rather than scalar indexing.
"""
function jac_diag_via_jvps(rec::TapedRecursion, x::AbstractVector, t::Int)
    D = length(x)
    FT = eltype(x)
    I_buf = similar(x, D, D)
    copyto!(I_buf, Matrix{FT}(I, D, D))
    d = similar(x)
    fill!(d, zero(FT))
    for i in 1:D
        ei = view(I_buf, :, i)
        jv = rec.jvp(x, rec.tape[t], ei)
        d .+= ei .* jv
    end
    return d
end

"""
Stochastic (Hutchinson/Rademacher) estimator of diag(J), where J is the Jacobian
of the surrogate step w.r.t. `x`.

    diag(J) ≈ (1/K) * Σ_k  z^(k) ⊙ (J z^(k)),   z_i ∈ {±1}

Keywords:
- `probes`: number of random probe vectors K (typical 1–4)
- `rng`: RNG for probes
- `zbuf`: optional preallocated buffer (length D) to avoid allocations
"""
function jac_diag_stoch(
    rec::TapedRecursion,
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

    d = fill!(similar(z, D), zero(FT))
    for _ in 1:probes
        _rademacher!(z, rng)
        jv = rec.jvp(x, rec.tape[t], z)
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

Keywords:
- `jacobian`: `:stoch_diag` (Hutchinson, 1 JVP per step — GPU default) or
  `:diag` (exact diagonal, D JVPs per step — useful for debugging on small problems).
- `damping`: in (0,1]
- `probes`: Hutchinson probes for `:stoch_diag`

When `rec.fwd_and_jvp` is set, the first (or only) Hutchinson probe fuses the
forward-step evaluation with the JVP, avoiding redundant primal `gradlogp` calls.
"""
function deer_update(
    rec::TapedRecursion,
    s0_in::AbstractVector,
    S::AbstractMatrix;
    jacobian::Symbol=:stoch_diag,
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

    A = similar(S)
    B = similar(S)
    zbuf = similar(s0, D)

    for t in 1:T
        xbar = t == 1 ? s0 : S[:, t - 1]

        if jacobian === :stoch_diag
            _rademacher!(zbuf, rng)

            # First probe: fuse forward pass and JVP when possible.
            ft, jvp1 = if rec.fwd_and_jvp !== nothing
                rec.fwd_and_jvp(xbar, rec.tape[t], zbuf)
            else
                (rec.step_fwd(xbar, rec.tape[t]), rec.jvp(xbar, rec.tape[t], zbuf))
            end
            jt = zbuf .* jvp1

            # Additional probes: only JVP needed (ft already computed above).
            for _ in 2:probes
                _rademacher!(zbuf, rng)
                jvp_k = rec.jvp(xbar, rec.tape[t], zbuf)
                jt .+= zbuf .* jvp_k
            end
            probes > 1 && (jt .*= one(eltype(jt)) / probes)

        elseif jacobian === :diag
            ft = rec.step_fwd(xbar, rec.tape[t])
            jt = jac_diag_via_jvps(rec, xbar, t)

        else
            throw(ArgumentError("jacobian must be :diag or :stoch_diag"))
        end

        view(A, :, t) .= jt
        view(B, :, t) .= ft .- jt .* xbar
    end

    S_new = solve_affine_scan_diag(A, B, s0)
    if damping != 1.0
        @. S_new = (1 - damping) * S + damping * S_new
    end
    return S_new
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
3. Pass a GPU vector as `initial_params`.

The `solve_affine_scan_diag` kernel runs as pure broadcasts and requires no
backend-specific GPU code; only the HVP computation inside the explicit JVP depends
on the chosen backend.

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
    jacobian::Symbol=:stoch_diag,
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
        S0 = similar(s0, D, T)
        S0 .= reshape(s0, D, 1)
        S0
    else
        size(init) == (D, T) || throw(ArgumentError("init must be size (D,T)"))
        copy(init)
    end

    converged = false
    last_metric = Inf
    iters = 0

    for iter in 1:maxiter
        iters = iter
        S_new = deer_update(
            rec, s0, S; jacobian=jacobian, damping=damping, probes=probes, rng=rng
        )
        
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
