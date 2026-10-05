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

The gradient argument may be either a callable or an `ADTypes.AbstractADType`
backend.

## Option 1: write the gradient

Hand-written derivatives keep the execution path predictable and are often the
best choice for GPU models.

```julia
using ParallelMCMC

logdensity(x) = -sum(abs2, x) / 2
gradient(x) = -x

model = DensityModel(
    logdensity,
    gradient,
    4;
    param_names=[:a, :b, :c, :d],
)
```

## Option 2: derive the gradient with AD

Pass an AD backend as the gradient argument to build a model from the
log-density alone:

```julia
using ADTypes, ForwardDiff, ParallelMCMC

logdensity(x) = -sum(abs2, x) / 2
model = DensityModel(logdensity, AutoForwardDiff(), 4)
```

Backends are prepared when sampling begins, once the package knows the element
type and storage of `initial_params`.

## Hessian-vector products for parallel MALA

Parallel MALA additionally requires Hessian-vector products (HVPs). Just like
for gradients, you can either hand-write them or use AD.

A hand-written HVP is passed with the `hvp` keyword:

```julia
hvp(x, v) = -v
model = DensityModel(logdensity, gradient, 4; hvp=hvp)
```

To use AD instead, provide a backend to the sampler:

```julia
model = DensityModel(logdensity, gradient, 4)
sampler = ParallelMALASampler(0.1; backend=AutoForwardDiff())
```

### Which function does an HVP backend differentiate?

| `gradient = ...` | `hvp = ...` | HVP calculated by... |
|---|---|---|
| callable | backend | differentiating `gradient` with `backend` |
| backend1 | backend2 | differentiating log-density with `SecondOrder(backend2, backend1)` |
| either | `SecondOrder(...)` | differentiating log-density with `SecondOrder(...)` |

Backend combinations are not equally useful on every device;
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

Both batched derivative arguments require `logdensity_batch`. When the
gradient argument is an AD backend, ParallelMCMC can derive a batched gradient by
differentiating `sum(logdensity_batch(X))`. For that to be correct, each output
of `logdensity_batch` must depend only on the corresponding column of `X`.

The scalar functions must still be provided even when all batched
functions are present.

## Parameter names

If `param_names` is not passed, the chain will contain one vector-valued
parameter named `:x`. Pass names when separate scalar or array-valued entries
would be easier to work with.

```julia
# Three scalar parameters
DensityModel(logdensity, gradient, 3; param_names=[:a, :b, :c])

# One scalar and one 1 x 2 matrix
DensityModel(logdensity, gradient, 3; param_names=[:a, (:B, (1, 2))])
```

Names may be `Symbol`s or `VarName`s. The declared shapes must account for
exactly `dim` scalar values. `VNChain` accepts `VarName` keys, while `SymChain`
uses `Symbol` keys.

## DynamicPPL models

DynamicPPL is the package that defines Turing's `@model` syntax. Loading it
activates a convenience constructor that handles unconstraining, parameter
names, and conversion back to the model's original parameter space. Unlike when
sampling with Turing, the AD backend must be explicitly passed, and is used to
calculate the gradient of the log-density.

```julia
using ADTypes, Distributions, DynamicPPL, FlexiChains, ForwardDiff, ParallelMCMC

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

DynamicPPL-backed models require `VNChain`. Their output includes `:logjoint`,
`:logprior`, and `:loglikelihood`; this is more informative than a manually
constructed model, which has a single `:logp` field.

To use parallel MALA, supply an analytical HVP or an appropriate AD backend:

```julia
using Enzyme

sampler = ParallelMALASampler(0.1; T=64, backend=AutoEnzyme())
chain = sample(model, sampler, 1_000; chain_type=VNChain)
```

You should not assume that a model's gradient backend is also a valid HVP
backend. In particular, applying ForwardDiff around a DynamicPPL gradient that
already uses ForwardDiff creates nested dual numbers, which are not supported.
For such a gradient, Enzyme is the supported default, but you may also specify
an explicit second-order backend pair or analytical HVP.

DynamicPPL models currently run on the CPU, because DynamicPPL evaluates the
model with CPU vectors. For a GPU target, write the density and derivatives
directly with GPU-compatible array operations.

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
