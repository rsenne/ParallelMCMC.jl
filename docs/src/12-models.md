# Defining models

Every sampler in ParallelMCMC works with a [`DensityModel`](@ref). The model is
the boundary between your statistical problem and the sampling algorithm: it
describes the target density, its derivatives, its dimension, and optionally
how parameters should be named in the result.

## The basic contract

The main constructor is:

```julia
DensityModel(logdensity, grad_logdensity, dim; kwargs...)
```

For a parameter vector `x` of length `dim`:

- `logdensity(x)` returns one real number. The density may be unnormalized.
- `grad_logdensity(x)` returns a vector with the same shape as `x`.
- `dim` is the number of unconstrained parameters.

The derivative slot may contain either a callable or an
`ADTypes.AbstractADType` backend.

## Option 1: write the derivatives

Hand-written derivatives keep the execution path predictable and are often the
best choice for GPU models.

```julia
using ParallelMCMC

logdensity(x) = -sum(abs2, x) / 2
gradient(x) = -x
hvp(x, v) = -v

model = DensityModel(
    logdensity,
    gradient,
    4;
    hvp=hvp,
    param_names=[:a, :b, :c, :d],
)
```

Sequential MALA needs the gradient. Parallel MALA also needs
Hessian-vector products (HVPs). You can supply `hvp(x, v)` as above, or let the
sampler construct it with an AD backend.

## Option 2: derive gradients with AD

Pass an AD backend in the gradient slot to build a model from the log-density
alone:

```julia
using ADTypes, ForwardDiff, ParallelMCMC

logdensity(x) = -sum(abs2, x) / 2
model = DensityModel(logdensity, AutoForwardDiff(), 4)
```

Backends are prepared when sampling begins, once the package knows the element
type and storage of `initial_params`.

For parallel MALA, also provide a backend to the sampler unless the model has
an `hvp`:

```julia
sampler = ParallelMALASampler(0.1; backend=AutoForwardDiff())
```

You can mix approaches. For example, keep a hand-written gradient and use AD
only for the HVP:

```julia
model = DensityModel(logdensity, gradient, 4)
sampler = ParallelMALASampler(0.1; backend=AutoForwardDiff())
```

### Which function does an HVP backend differentiate?

| Gradient slot | HVP slot | What ParallelMCMC differentiates |
|---|---|---|
| callable | backend | the callable gradient |
| backend | backend | the log-density with a second-order backend pair |
| either | `SecondOrder(...)` | the log-density with that explicit pair |

An explicit `DifferentiationInterface.SecondOrder` takes precedence over the
gradient slot. Backend combinations are not equally useful on every device;
see [GPU execution](15-gpu.md) before selecting one for GPU work.

## Batched functions

DEER evaluates corresponding operations across all `T` columns of a trajectory.
If your model has an efficient matrix implementation, provide:

```julia
logdensity_batch(X)             # one log-density per column
grad_logdensity_batch(X)        # one gradient per column
hvp_batch(X, V)                 # H(X[:, j]) * V[:, j] per column
```

Pass them as keywords:

```julia
model = DensityModel(
    logdensity,
    gradient,
    dim;
    hvp=hvp,
    logdensity_batch=logdensity_batch,
    grad_logdensity_batch=grad_logdensity_batch,
    hvp_batch=hvp_batch,
)
```

Both batched derivative slots require `logdensity_batch`. When the gradient
slot contains an AD backend, ParallelMCMC can derive a batched gradient by
differentiating `sum(logdensity_batch(X))`. For that to be correct, each output
of `logdensity_batch` must depend only on the corresponding column of `X`.

The scalar functions remain part of the contract even when all batched
functions are present.

## Parameter names

Without `param_names`, the chain contains one vector-valued parameter named
`:x`. Pass names when separate scalar or array-valued entries would be easier to
work with.

```julia
# Three scalar parameters
DensityModel(logdensity, gradient, 3; param_names=[:a, :b, :c])

# One scalar and one 1 x 2 matrix
DensityModel(logdensity, gradient, 3; param_names=[:a, (:B, (1, 2))])
```

Names may be `Symbol`s or `VarName`s. The declared shapes must account for
exactly `dim` scalar values. `VNChain` accepts `VarName` keys, while `SymChain`
uses `Symbol` keys.

## Turing models

Loading Turing activates a convenience constructor that handles unconstraining,
parameter names, and conversion back to the model's original parameter space.
The AD backend is required.

```julia
using ADTypes, FlexiChains, ForwardDiff, ParallelMCMC, Turing

@model function normal_location(y)
    mu ~ Normal(0, 1)
    y ~ Normal(mu, 0.5)
end

model = DensityModel(
    normal_location(1.5);
    ad_backend=AutoForwardDiff(),
)

chain = sample(
    model,
    AdaptiveMALASampler(0.3; n_warmup=500),
    2_000;
    chain_type=VNChain,
    discard_warmup=true,
)
```

Turing-backed models require `VNChain`. Their output includes `:logjoint`,
`:logprior`, and `:loglikelihood`; a manually constructed model has a single
`:logp` field.

To use parallel MALA, supply an HVP or an appropriate sampler backend:

```julia
using Enzyme

sampler = ParallelMALASampler(0.1; T=64, backend=AutoEnzyme())
chain = sample(model, sampler, 1_000; chain_type=VNChain)
```

Do not assume that the model's gradient backend is also a valid HVP backend.
In particular, applying ForwardDiff around a DynamicPPL gradient that already
uses ForwardDiff creates unsupported nested dual numbers. Enzyme is the
supported default for differentiating that gradient; an explicit second-order
backend pair or analytical HVP can also be used.

Turing models currently run on the CPU. For a GPU target, write the density and
derivatives directly with GPU-compatible array operations.

## LogDensityProblems models

Any object implementing the gradient-capable
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface
can be wrapped directly:

```julia
model = DensityModel(logdensity_problem; param_names=[:a, :b])
```

This constructor gets the dimension, log-density, and gradient from the
interface. The same HVP and batching keywords accepted by the main constructor
are available when you need them.
