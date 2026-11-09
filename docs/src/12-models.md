# Defining models

Every sampler in ParallelMCMC samples from a [`DensityModel`](@ref).
A `DensityModel` holds the log-density, its derivatives, the number of parameters, and optionally the parameter names used in the returned chain.

## The basic contract

The main constructor is:

```julia
DensityModel(logdensity, grad_logdensity, dim; kwargs...)
```

For a parameter vector `x` of length `dim`:

- `logdensity(x)` returns one real number. The density may be unnormalized.
- `grad_logdensity(x)` returns a vector with the same shape as `x`.
- `dim` is the number of unconstrained parameters.

The gradient argument may be either a function or an `ADTypes.AbstractADType` backend.

## Option 1: write the gradient

A hand-written gradient needs no AD package and has no AD compilation step.
On the GPU it also avoids the backend restrictions described in [GPU execution](15-gpu.md).

```@example models
using ParallelMCMC

logdensity(x) = -sum(abs2, x) / 2
gradient(x) = -x

model = DensityModel(logdensity, gradient, 4; param_names=[:a, :b, :c, :d])
nothing # hide
```

## Option 2: derive the gradient with AD

Pass an AD backend as the gradient argument to build a model from the log-density alone:

```@example models
using ADTypes, ForwardDiff

model = DensityModel(logdensity, AutoForwardDiff(), 4)
nothing # hide
```

ParallelMCMC prepares the backend when `sample` starts, using the element type and array type of `initial_params`.
The first call to `sample` therefore includes the AD preparation time.

## Hessian-vector products for parallel MALA

[`ParallelMALASampler`](@ref) also needs Hessian-vector products (HVPs) $\nabla^2 \log p(x)\, v$.
The sequential samplers do not use them.
As with gradients, you can write the HVP yourself or let an AD backend compute it.

A hand-written HVP is passed with the `hvp` keyword:

```@example models
hvp(x, v) = -v
model = DensityModel(logdensity, gradient, 4; hvp=hvp)
nothing # hide
```

To compute the HVP with AD, pass a backend either as the model's `hvp` or as the sampler's `backend`:

```@example models
model = DensityModel(logdensity, gradient, 4)
sampler = ParallelMALASampler(0.1; backend=AutoForwardDiff())
nothing # hide
```

If the gradient argument is already an AD backend, as in Option 2, you can leave out both.
The sampler then builds the HVP from the gradient's backend, as the table below shows.

### How the HVP is computed

The HVP backend is the model's `hvp` if it is a backend, and otherwise the sampler's `backend`.

| Gradient argument | HVP backend | How the HVP is computed |
|---|---|---|
| function | `b` | `b` differentiates the gradient function |
| backend `g` | `b` | `SecondOrder(b, g)` differentiates the log-density |
| either | `SecondOrder(...)` | that `SecondOrder` differentiates the log-density |
| backend `g` | none | `SecondOrder(forward_counterpart(g), g)` differentiates the log-density |
| function | none | `sample` throws an `ArgumentError` |

`SecondOrder` and `forward_counterpart` come from DifferentiationInterface.
`forward_counterpart` returns a forward-mode version of the backend when one exists, such as `AutoMooncakeForward` for `AutoMooncake`, and the backend itself otherwise.

Not every combination works on every device.
See [GPU execution](15-gpu.md) before choosing one for GPU work.

### AD or a hand-written HVP?

AD is the simpler choice when the Hessian has no convenient closed form, as in hierarchical models with many transformations, mixtures, or neural networks.
It also lets you switch backends without changing the model.

A hand-written HVP is worth writing when it has a closed form in quantities the gradient already computes, as in generalized linear models.
It skips AD compilation, which matters when you run many short chains.
On the GPU it also avoids the Enzyme restrictions in [GPU execution](15-gpu.md), because the sampler then calls no AD backend.

## Batched functions

Each DEER iteration evaluates the log-density, the gradient, and HVPs at all `T` states of the current trajectory.
By default it does this one state at a time.
If your model can process a matrix whose columns are states, pass batched versions:

```@example models
logdensity_batch(X) = vec(-sum(abs2, X; dims=1) / 2)  # one value per column
grad_logdensity_batch(X) = -X                          # column j is the gradient at X[:, j]
hvp_batch(X, V) = -V                                   # column j is H(X[:, j]) * V[:, j]

model = DensityModel(
    logdensity,
    gradient,
    4;
    hvp=hvp,
    logdensity_batch=logdensity_batch,
    grad_logdensity_batch=grad_logdensity_batch,
    hvp_batch=hvp_batch,
)
nothing # hide
```

Only [`ParallelMALASampler`](@ref) uses the batched derivatives, and only with `jacobian=:stoch_diag`, so you still need the scalar functions.
`grad_logdensity_batch` and `hvp_batch` both require `logdensity_batch`.

Like `hvp`, `hvp_batch` can be a function or an AD backend.
If you leave it out, the sampler picks a backend for it by the same rules as for `hvp`.
A hand-written scalar `hvp` is not used for the batched HVP, so a model with batched functions needs `hvp_batch`, an HVP backend, or an AD gradient.

If the gradient argument is an AD backend and you leave out `grad_logdensity_batch`, ParallelMCMC differentiates `sum(logdensity_batch(X))` instead.
This is correct only if each entry of `logdensity_batch(X)` depends on the matching column of `X` alone.

## Parameter names

If `param_names` is not passed, the chain contains one vector-valued parameter named `:x`.
Pass names when separate scalar or array-valued entries are easier to work with.

```julia
# Three scalar parameters
DensityModel(logdensity, gradient, 3; param_names=[:a, :b, :c])

# One scalar and one 1 x 2 matrix
DensityModel(logdensity, gradient, 3; param_names=[:a, :B => (1, 2)])
```

Names may be `Symbol`s or `VarName`s.
Give an array-valued entry as a pair of its name and shape, as with `:B => (1, 2)` above.
The shapes must add up to exactly `dim` values.
The default chain type, `VNChain`, converts `Symbol` names to `VarName`s.
Pass `chain_type=SymChain` to `sample` to keep `Symbol` keys.

## LogDensityProblems models

You can wrap any object that implements the [LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface with gradients.
The constructor becomes available once you load LogDensityProblems:

```julia-repl
pkg> add LogDensityProblems
```

```@example ldp
using LogDensityProblems, ParallelMCMC

# mu ~ Normal(0, 1), y ~ Normal(mu, 0.5)
struct NormalPosterior
    y::Float64
end

LogDensityProblems.capabilities(::Type{NormalPosterior}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(::NormalPosterior) = 1
LogDensityProblems.logdensity(ld::NormalPosterior, x) = -x[1]^2 / 2 - 2 * (ld.y - x[1])^2

function LogDensityProblems.logdensity_and_gradient(ld::NormalPosterior, x)
    return LogDensityProblems.logdensity(ld, x), [-x[1] + 4 * (ld.y - x[1])]
end

model = DensityModel(NormalPosterior(1.5); param_names=[:mu])
chain = sample(model, AdaptiveMALASampler(0.3; n_warmup=500), 2_000; discard_warmup=true)
```

The constructor reads the dimension, log-density, and gradient from the interface.
Your type must implement `capabilities` (returning at least `LogDensityOrder{1}`), `dimension`, `logdensity`, and `logdensity_and_gradient`.
It accepts the same HVP and batching keywords as the main constructor.

These models have no default HVP, so for parallel MALA you have to pass `hvp` or a sampler `backend`.
An AD backend differentiates `logdensity_and_gradient`, which must therefore be differentiable by that backend.
If it is not, for example because it already calls AD internally, pass a `DifferentiationInterface.SecondOrder` backend instead, which differentiates `logdensity` twice.
`AutoReactant()` cannot be used with these models.
