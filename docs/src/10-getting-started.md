# Getting started

This guide takes a model from a log-density to a tuned parallel MALA run. The
example uses a banana-shaped target because it is nonlinear enough to exercise
the derivative path while still being easy to inspect.

## Install an AD backend

Parallel MALA is the main algorithm in ParallelMCMC. It requires an automatic
differentiation backend to compute Hessian–vector products, so you will have to
load one yourself. In this guide we will use ForwardDiff.

```julia-repl
pkg> add ParallelMCMC ForwardDiff
```

```julia
using AbstractMCMC: MCMCThreads
using ADTypes, FlexiChains, ForwardDiff, ParallelMCMC, Random
```

## Define the target

A [`DensityModel`](@ref) needs a log-density, a gradient, and the number of
unconstrained parameters. The log-density may omit its normalizing constant.

```julia
function banana_logdensity(x)
    a = x[2] + 0.1 * x[1]^2 - 1
    return -(x[1]^2 / 100 + a^2) / 2
end

function banana_gradient(x)
    a = x[2] + 0.1 * x[1]^2 - 1
    return [-x[1] / 100 - 0.2 * x[1] * a, -a]
end

model = DensityModel(
    banana_logdensity,
    banana_gradient,
    2;
    param_names=[:x1, :x2],
)
```

Here, the gradient is hand-written, but it does not have to be. See
[Defining models](12-models.md) for AD-derived gradients and analytical
Hessian-vector products.

## Tune the step size

MALA is sensitive to its step size, such that a value that is too large causes
frequent rejections, while one that is too small produces highly correlated
samples. Start with [`AdaptiveMALASampler`](@ref), which tunes the step size
during warmup.

```julia
rng = MersenneTwister(42)
tuning = sample(
    rng,
    model,
    AdaptiveMALASampler(0.1; n_warmup=500),
    750;
    initial_params=zeros(2),
    chain_type=VNChain,
    discard_warmup=true,
)

epsilon = last(vec(tuning[:step_size]))
```

The requested 750 transitions include warmup; `discard_warmup=true` removes
every transition marked as warmup from the returned chain. `:step_size`,
`:accepted`, and `:logp` are stored as extra fields alongside the parameters.

!!! tip "Treat tuning as a diagnostic"
    Check that the tuned step size settles to a finite, positive value and that
    the chain moves through the target. Automatic tuning cannot rescue an
    incorrect gradient or a badly chosen parameterization.

## Run parallel MALA

Pass the tuned value to [`ParallelMALASampler`](@ref).

```julia
sampler = ParallelMALASampler(
    epsilon;
    T=64,
    backend=AutoForwardDiff(),
)

chain = sample(
    MersenneTwister(43),
    model,
    sampler,
    2_000;
    initial_params=zeros(2),
    chain_type=VNChain,
)
```

The important arguments are:

- `epsilon`: the MALA step size.
- `T`: the number of transitions in each DEER solve. It does not change the
  number of samples returned by `sample`.
- `backend`: the AD backend used for missing Hessian-vector products. Leave it
  out when the model provides `hvp` and, when applicable, `hvp_batch`.
- `initial_params`: the first state.
- `chain_type`: use `VNChain` for `VarName` keys or `SymChain` for `Symbol`
  keys. Turing models require `VNChain`.

## Choose `T` and the DEER controls

`T=64` is a reasonable first experiment. Larger blocks expose more work to
parallel hardware but use more memory and can make the nonlinear solve harder.
Increase `T` only after the model works at a smaller value.

The defaults are intended to be a usable starting point:

```julia
ParallelMALASampler(
    epsilon;
    T=64,
    jacobian=:stoch_diag,
    probes=1,
    damping=0.5,
    maxiter=200,
    tol_abs=1e-6,
    tol_rel=1e-5,
    backend=AutoForwardDiff(),
)
```

- `jacobian=:stoch_diag` uses a Hutchinson estimate of the Jacobian diagonal.
  This is the scalable default.
- `probes` controls the number of random vectors in that estimate. More probes
  reduce estimator noise but require more derivative work.
- `damping` blends each DEER update with the previous trajectory. Lower it when
  the solve is unstable; raise it cautiously when convergence is already
  reliable.
- `maxiter`, `tol_abs`, and `tol_rel` control the nonlinear solve. Tightening
  tolerances costs more iterations.

For a low-dimensional correctness check, `jacobian=:diag` computes the exact
diagonal with one JVP per dimension. It is usually too expensive for large
models.

## Use the returned chain

ParallelMCMC returns a [FlexiChains](https://pysm.dev/FlexiChains.jl/) chain.
With the parameter names above, values are available as `chain[:x1]` and
`chain[:x2]`. Sampler diagnostics are stored in extra columns such as
`chain[:accepted]` and `chain[:logp]`.

Use several chains for convergence diagnostics. Start Julia with multiple
threads, for example `julia -t 4`, then run:

```julia
chains = sample(
    model,
    sampler,
    MCMCThreads(),
    2_000,
    4;
    initial_params=zeros(2),
    chain_type=VNChain,
)

ess(chains)
```

`MCMCThreads()` runs independent chains concurrently, which is separate from
the within-chain parallelism controlled by `T`.
