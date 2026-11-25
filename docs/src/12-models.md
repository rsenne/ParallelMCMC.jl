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

If we would rather not use a hand written gradient, we can pass an AD backend as the gradient argument to build a model from the log-density alone:

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

!!! warning "Second-order AD on the GPU"
    The rows with `SecondOrder` differentiate the log-density twice.
    On the GPU, only Zygote and Reactant managed this in our tests, and Zygote's first `sample` call took about 100 s.
    With a hand-written gradient the sampler differentiates only once, and ForwardDiff, Zygote, and Mooncake all work on `CuArray`s.
    See [GPU execution](15-gpu.md#HVPs-from-an-AD-backend) for what we tested.

### AD or a hand-written HVP?

AD is the simpler choice when the Hessian has no convenient closed form, as in hierarchical models with many transformations, mixtures, or neural networks.
It also lets you switch backends without changing the model.

A hand-written HVP is worth writing when it has a closed form in quantities the gradient already computes, as in generalized linear models.
On the GPU it also avoids the Enzyme restrictions in [GPU execution](15-gpu.md), because the sampler then calls no AD backend.

## Batched functions

Each DEER iteration evaluates the log-density, the gradient, and HVPs at all `T` states of the current trajectory.
By default it does this one state at a time.
If your model can process a matrix whose columns are states, you can pass batched versions:

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

Recall that each DEER iteration takes one MALA step from every state in the trajectory.
That step needs the log-density for the accept/reject test, the gradient for the proposal, and an HVP for the Jacobian.
The batched path computes all three on the whole $D \times T$ matrix, so it needs a batched version of each.
This is why `grad_logdensity_batch` and `hvp_batch` both require `logdensity_batch`.

Like `hvp`, the batched HVP can be your own function or come from an AD backend, picked by the same rules.
ParallelMCMC will not loop your scalar `hvp` over the columns for you, though, so a model with batched functions needs `hvp_batch`, an HVP backend, or an AD gradient with no `grad_logdensity_batch`.

However, you still need the scalar functions.
The sequential samplers use them, `jacobian=:diag` takes the unbatched path, and the sampler evaluates the starting point on its own.

If the gradient is an AD backend and you leave out `grad_logdensity_batch`, ParallelMCMC differentiates `sum(logdensity_batch(X))` to get the batched gradient.
Column $j$ of that gradient is the gradient of entry $j$ only when no other entry depends on column $j$.
That holds when `logdensity_batch` evaluates each column on its own, but not if it mixes columns, for example by subtracting their mean.

### Value, gradient, and HVP in one call

Each DEER iteration evaluates the log-density, gradient, and HVP at the current states and at the proposals.
These often share most of their work.
In a generalized linear model, for example, each of those six calls recomputes $X B$.
`value_grad_hvp_batch` takes one function that returns all three, so the shared work runs once per state.
For the logistic regression in [GPU execution](15-gpu.md), it looks like this:

```julia
function value_grad_hvp_batch(B, V)
    Z  = X * B
    S  = sigmoid.(Z)
    lp = vec(sum(y .* Z .- softplus.(Z); dims=1)) .- vec(sum(abs2, B; dims=1)) ./ 2
    G  = X' * (y .- S) .- B
    Hv = -(X' * (S .* (1 .- S) .* (X * V))) .- V
    return lp, G, Hv
end

model = DensityModel(logp, gradlogp, D;
                     hvp=hvp,
                     logdensity_batch=logp_batch,
                     value_grad_hvp_batch=value_grad_hvp_batch)
```

It replaces `grad_logdensity_batch` and `hvp_batch`, so the model cannot have those as well.
`logdensity_batch` and a scalar HVP are still needed.
The sampler copies the three results before its next call, so the function may return buffers that it reuses.
With `jacobian=:diag`, DEER takes the unbatched path and does not call it.

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
To keep `Symbol` keys, load FlexiChains and pass `chain_type=SymChain` to `sample`.

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

To ParallelMCMC, a LogDensityProblems model looks like one with a hand-written gradient, so it falls in the "function, none" row of the table above.
There is no default HVP, and for parallel MALA you have to pass `hvp` or a sampler `backend`.
An AD backend differentiates `logdensity_and_gradient`, which must therefore be differentiable by that backend.
If it is not, for example because it already calls AD internally, pass a `DifferentiationInterface.SecondOrder` backend instead, which differentiates `logdensity` twice.
`AutoReactant()` cannot be used with these models.
