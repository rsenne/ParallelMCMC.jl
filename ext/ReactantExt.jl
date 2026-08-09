module ReactantExt

#=
Reactant-compiled derivative paths, selected by `ADTypes.AutoReactant()` in
any derivative slot of `DensityModel` (or as the sampler `backend`).
Derivatives are traced with Enzyme-MLIR and compiled to XLA executables via
`Reactant.@compile`, bypassing Enzyme's LLVM pipeline — and with it the GPU
`cuMemcpyDtoHAsync_v2` gc-transition abort. This is the only path that
computes a genuine second-order HVP on GPU.

Requirements / limitations:
  - The traced function (`logdensity` / `gradlogp` / batched forms) must be
    Reactant-traceable: plain array ops. DynamicPPL-built log-densities do
    NOT trace as-is.
  - Executables are shape-specialized to the preparation templates, and
    arguments are marshalled to/from Reactant's own (XLA) device memory on
    every call. A boundary copy, not a fused in-place path — a target for
    later optimization.
  - HVPs are explicit forward-over-(gradient) compositions; NEVER
    `Enzyme.hvp`, which silently returns zeros under `@compile`.
  - `AutoReactant.mode` (the wrapped `AutoEnzyme`) is currently ignored:
    gradients always trace as Enzyme reverse, HVPs as forward-over-that.
=#

using ParallelMCMC: ParallelMCMC
using ParallelMCMC.DEER: DEER
using ADTypes: AutoReactant
using Reactant: Reactant, @compile
using Enzyme: Enzyme

# Host-materialize for the Reactant boundary: Reactant manages its own (XLA)
# device memory, so we round-trip through a plain host Array regardless of
# where the package's arrays live (Vector / CuArray).
_host(x::Array) = x
_host(x::AbstractArray) = Array(x)

function _from_host(template::AbstractArray, out)
    res = similar(template, size(out))
    copyto!(res, Array(out))
    return res
end

#=
Compile `core` for the template shapes once and return a closure that
marshals package arrays <-> Reactant arrays.
=#
function _compiled(core, t1::AbstractArray)
    r1 = Reactant.to_rarray(_host(t1))
    compiled = @compile core(r1)
    return x -> _from_host(x, compiled(Reactant.to_rarray(_host(x))))
end

function _compiled(core, t1::AbstractArray, t2::AbstractArray)
    r1 = Reactant.to_rarray(_host(t1))
    r2 = Reactant.to_rarray(_host(t2))
    compiled = @compile core(r1, r2)
    return function (x, v)
        out = compiled(Reactant.to_rarray(_host(x)), Reactant.to_rarray(_host(v)))
        return _from_host(x, out)
    end
end

# Forward-mode JVP of `g` in direction `v`, i.e. J(g)·v. For g = gradlogp
# this is the HVP H·v. `Const(g)` so Enzyme doesn't treat captures as active.
function _jvp(g, x, v)
    return only(Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(g), Enzyme.Duplicated(x, v)))
end

_rev_gradient(f, x) = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), x)[1]

# Columns of X are independent samples, so ∇_X sum(logp_batch(X)) stacks the
# per-column gradients.
_sumbatch(f, X) = sum(f(X))

#=
Gradient slots. The wrappers keep the raw log-density so the HVP factories
below can re-trace forward-over-reverse from it, instead of trying to trace
through an already-compiled executable.
=#
struct _ReactantGradient{F,C}
    logdensity::F
    compiled::C
end
(g::_ReactantGradient)(x) = g.compiled(x)

function ParallelMCMC._reactant_resolve_gradient(
    logdensity, backend::AutoReactant, x_template::AbstractVector
)
    core = Base.Fix1(_rev_gradient, logdensity)
    return _ReactantGradient(logdensity, _compiled(core, x_template))
end

struct _ReactantGradientBatch{F,C}
    logdensity_batch::F
    compiled::C
end
(g::_ReactantGradientBatch)(X) = g.compiled(X)

function ParallelMCMC._reactant_resolve_gradient_batch(
    logdensity_batch, backend::AutoReactant, X_template::AbstractMatrix
)
    core = Base.Fix1(_rev_gradient, Base.Fix1(_sumbatch, logdensity_batch))
    return _ReactantGradientBatch(logdensity_batch, _compiled(core, X_template))
end

#=
HVP factories. Two tracings depending on where the gradient came from:

  - `_ReactantGradient` (the gradient slot was itself AutoReactant):
    re-trace from the raw log-density as explicit forward-over-reverse —
    genuine second-order AD fused into one XLA program.
  - any other callable (hand-written gradient): forward JVP over it,
    provided it is traceable.
=#
function DEER._make_hvp_fn(
    ::DEER.ReactantHVP,
    gradlogp::_ReactantGradient,
    backend::AutoReactant,
    x_template::AbstractVector,
)
    inner = Base.Fix1(_rev_gradient, gradlogp.logdensity)
    core(x, v) = _jvp(inner, x, v)
    return _compiled(core, x_template, x_template)
end

function DEER._make_hvp_fn(
    ::DEER.ReactantHVP, gradlogp, backend::AutoReactant, x_template::AbstractVector
)
    core(x, v) = _jvp(gradlogp, x, v)
    return _compiled(core, x_template, x_template)
end

function DEER._make_hvp_batch_fn(
    ::DEER.ReactantHVP,
    grad_batch::_ReactantGradientBatch,
    backend::AutoReactant,
    X_template::AbstractMatrix,
)
    inner = Base.Fix1(_rev_gradient, Base.Fix1(_sumbatch, grad_batch.logdensity_batch))
    core(X, V) = _jvp(inner, X, V)
    return _compiled(core, X_template, X_template)
end

function DEER._make_hvp_batch_fn(
    ::DEER.ReactantHVP, grad_batch, backend::AutoReactant, X_template::AbstractMatrix
)
    # Column-independent batched gradient ⇒ forward JVP is the columnwise HVP.
    core(X, V) = _jvp(grad_batch, X, V)
    return _compiled(core, X_template, X_template)
end

end # module
