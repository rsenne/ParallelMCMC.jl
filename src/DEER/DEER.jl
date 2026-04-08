module DEER

using LinearAlgebra
using DifferentiationInterface
using ADTypes: ADTypes, AbstractADType
import Enzyme: Enzyme
using Random

include("DEERScan.jl")
using .DEERScan

const DI = DifferentiationInterface
# Enzyme forward-mode is optimal for JVPs used in :stoch_diag.
# NOTE: On Windows, Enzyme CPU has a known LLVM unwinding issue (pinned v0.13.131).
# For Windows CPU use, construct samplers with backend=AutoForwardDiff() explicitly.
const DEFAULT_BACKEND = ADTypes.AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Duplicated)

"""
Deterministic recursion driven by a pre-generated tape.

- `step_fwd(x, tape_t)`: exact forward transition (may include MH accept/reject).
- `jvp(x, tape_t, v)`: explicit JVP `(x, tape_t, v) -> J·v` of the differentiable
  surrogate step.  Required; must not call AD through `step_fwd`'s control flow.
- `fwd_and_jvp(x, tape_t, v)`: optional fused function `-> (x_next, J·v)` that
  shares primal computation between the forward step and the JVP.  When provided,
  `deer_update` uses it for the first (or only) Hutchinson probe, halving the number
  of `gradlogp` and `logp` evaluations per time step.  Defaults to `nothing`.
- `fwd_and_jvp_batch(Xbar, Z)`: optional batched function `-> (FT, Jt)` that
  processes all T time steps simultaneously (GPU-efficient).  When provided,
  `deer_update` uses this path instead of the sequential loop for `:stoch_diag`.
  `Xbar` is D×T (previous states), `Z` is D×T (Rademacher probes), returns
  `FT` (D×T, exact forward steps) and `Jt` (D×T, Hutchinson diagonal estimates).
  Defaults to `nothing`.
"""
struct TapedRecursion{Ff,Fj,Ffj,Ffb,Tt}
    step_fwd::Ff        # (x, tape_t) -> x_next
    jvp::Fj             # (x, tape_t, v) -> J·v  (explicit, required)
    fwd_and_jvp::Ffj    # (x, tape_t, v) -> (x_next, J·v), or Nothing
    fwd_and_jvp_batch::Ffb  # (Xbar::D×T, Z::D×T) -> (FT::D×T, Jt::D×T), or Nothing
    tape::Vector{Tt}
end

"Main constructor."
function TapedRecursion(
    step_fwd,
    jvp,
    tape::Vector;
    fwd_and_jvp=nothing,
    fwd_and_jvp_batch=nothing,
)
    return TapedRecursion(step_fwd, jvp, fwd_and_jvp, fwd_and_jvp_batch, tape)
end

"""
Prepare a reusable pushforward object for HVP/JVP evaluations of `f`.

The returned preparation object is intended to be reused across repeated calls
with the same primal/tangent types, shape, and device placement as `x_template`.
"""
function _prepare_hvp(
    f,
    backend::AbstractADType,
    x_template::AbstractVector,
)
    v_template = similar(x_template)
    fill!(v_template, zero(eltype(x_template)))
    return DI.prepare_pushforward(f, backend, x_template, (v_template,))
end

"""
Use a precomputed pushforward preparation object to evaluate one HVP/JVP.

This is the hot-path alternative to `_hvp_nopre`, avoiding repeated
`prepare_pushforward` calls.
"""
function _hvp_prepared(
    f,
    prep,
    backend::AbstractADType,
    x::AbstractVector,
    v::AbstractVector,
)
    res = DI.pushforward(f, prep, backend, x, (v,))
    return res isa Tuple ? first(res) : res
end

"""
Compute a single JVP of `f` at `x` in direction `v` using `backend`, preparing
internally on every call.  Used as a fallback when no reusable prepared object
is available.
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
Fill the D×T matrix `Z` with ±1 Rademacher entries.

Generates the full D×T matrix on CPU then transfers in a single
`copyto!`, avoiding T separate host-to-device copies.
GPU-compatible: `copyto!(CuMatrix, Matrix)` is a single H→D transfer.
"""
@inline function _rademacher_matrix!(Z::AbstractMatrix{T}, rng::AbstractRNG) where {T}
    D, n = size(Z)
    bits = rand(rng, Bool, D, n)
    vals = Matrix{T}(undef, D, n)
    @. vals = ifelse(bits, one(T), -one(T))
    copyto!(Z, vals)
    return Z
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
Compute one DEER update given a trajectory guess `S` (D×T).
Returns `S_new` (D×T).

Keywords:
- `jacobian`: `:stoch_diag` (Hutchinson, 1 JVP per step — GPU default) or
  `:diag` (exact diagonal, D JVPs per step — useful for debugging on small problems).
- `damping`: in (0,1]
- `probes`: Hutchinson probes for `:stoch_diag` (sequential path only)

**Batched path** (GPU-efficient): when `rec.fwd_and_jvp_batch !== nothing` and
`jacobian === :stoch_diag`, all T time steps are processed simultaneously using
matrix operations.  A single D×T Rademacher matrix replaces T separate
host-to-device transfers.

**Sequential path** (fallback): used when no batch function is available or
`jacobian === :diag`.  When `rec.fwd_and_jvp` is set, the first (or only)
Hutchinson probe fuses the forward-step evaluation with the JVP.
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

    if rec.fwd_and_jvp_batch !== nothing && jacobian === :stoch_diag
        # Build D×T matrix of previous states: Xbar[:,1] = s0, Xbar[:,t] = S[:,t-1]
        Xbar = similar(S)
        copyto!(view(Xbar, :, 1), s0)
        T > 1 && (Xbar[:, 2:end] .= S[:, 1:(T - 1)])

        # Generate all T Rademacher probes in one host-to-device transfer.
        Z = similar(S)
        _rademacher_matrix!(Z, rng)

        FT, Jt = rec.fwd_and_jvp_batch(Xbar, Z)
        A .= Jt
        B .= FT .- Jt .* Xbar

    else
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
        metric ≤ 1 && (converged = true; break)
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