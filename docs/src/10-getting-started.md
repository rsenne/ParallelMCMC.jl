# Getting started

This guide samples a two-dimensional banana-shaped density with parallel MALA.
We tune the step size with sequential MALA first, then reuse it in the parallel sampler.
The target is

```math
\log p(x) = -\frac{1}{2}\left(\frac{x_1^2}{100} + \left(x_2 + 0.1\, x_1^2 - 1\right)^2\right).
```

Its Hessian changes with $x$, so it tests the Hessian-vector products in a way that a Gaussian target would not.

## Installation

Parallel MALA is the main algorithm in ParallelMCMC.
It needs Hessian-vector products of the log-density, which you can write yourself or get from an automatic differentiation (AD) backend.
ParallelMCMC does not depend on any AD package, so you load the one you want.
This guide uses ForwardDiff.

```julia-repl
pkg> add ParallelMCMC ADTypes ForwardDiff FlexiChains AbstractMCMC
```

ADTypes provides `AutoForwardDiff()`, FlexiChains provides `ess` and `rhat`, and AbstractMCMC provides `MCMCThreads()`.

```@example gs
using ParallelMCMC
using ADTypes, FlexiChains, ForwardDiff, Random
using AbstractMCMC: MCMCThreads
```

## Define the target

A [`DensityModel`](@ref) needs a log-density, a gradient, and the number of unconstrained parameters.
The log-density may omit its normalizing constant.

```@example gs
function banana_logdensity(x)
    a = x[2] + 0.1 * x[1]^2 - 1
    return -(x[1]^2 / 100 + a^2) / 2
end

function banana_gradient(x)
    a = x[2] + 0.1 * x[1]^2 - 1
    return [-x[1] / 100 - 0.2 * x[1] * a, -a]
end

model = DensityModel(banana_logdensity, banana_gradient, 2; param_names=[:x1, :x2])
nothing # hide
```

The gradient here is written by hand, but it does not have to be.
[Defining models](12-models.md) shows how to get it from an AD backend instead.

## Tune the step size

MALA is sensitive to its step size.
If the step size is too large, most proposals are rejected; if it is too small, the chain moves slowly and successive samples are highly correlated.
[`AdaptiveMALASampler`](@ref) tunes the step size during warmup, so we run it first and reuse the result.

```@example gs
tuning = sample(
    MersenneTwister(42),
    model,
    AdaptiveMALASampler(0.1; n_warmup=500),
    750;
    initial_params=zeros(2),
    discard_warmup=true,
)

epsilon = last(vec(tuning[:step_size]))
```

The 750 transitions include the 500 warmup transitions, and `discard_warmup=true` removes those from the returned chain.
After warmup the step size is fixed, so we take the last value of `:step_size`, which the chain stores next to the parameters.

!!! tip "Check the tuned step size"
    A tuned step size that collapses toward zero usually means that the gradient is wrong or that the parameters have very different scales.
    Adaptation cannot fix either problem.

## Run parallel MALA

Pass the tuned step size to [`ParallelMALASampler`](@ref).
The model has no hand-written Hessian-vector product, so `backend=AutoForwardDiff()` tells the sampler to compute one by differentiating `banana_gradient`.

```@example gs
sampler = ParallelMALASampler(epsilon; T=64, backend=AutoForwardDiff())
chain = sample(MersenneTwister(43), model, sampler, 2_000; initial_params=zeros(2))
```

`T=64` sets how many transitions DEER solves together; `sample` still returns 2,000 samples.

## Choose `T` and the DEER controls

Start with the default `T=64`, and change it only after the model samples correctly.
Larger blocks give parallel hardware more work per iteration, but they use more memory and the nonlinear solve may need more iterations.

The other keywords control the solve:

- `jacobian=:stoch_diag`, the default, estimates the diagonal of each transition's Jacobian with `probes` random vectors. Each extra probe makes the estimate less noisy and costs one more Jacobian-vector product per transition.
- `jacobian=:diag` computes the diagonal exactly with one Jacobian-vector product per parameter. Use it to check results on small models.
- `jacobian_damping` and `jacobian_clip` scale and clip the Jacobian diagonal before each scan. The defaults suit most targets; try `jacobian_damping=0.5` if the solve needs many iterations or its iterates blow up. See [Damping and clipping the Jacobian](20-algorithms.md#Damping-and-clipping-the-Jacobian).
- `maxiter`, `tol_abs`, and `tol_rel` decide when the solve stops. See [`ParallelMALASampler`](@ref) for the exact rule.

If a block reaches `maxiter` before it meets the tolerances, the sampler still returns its states as samples and logs one warning for the run.
[Troubleshooting](16-troubleshooting.md) explains what to try.

## Use the returned chain

`sample` returns a [FlexiChains](https://pysm.dev/FlexiChains.jl/) chain.
Index it with the parameter names from the model, as in `chain[:x1]`, and use `chain[:logp]` to get the log-density of each sample.

## Run several chains

To check convergence, run several chains and compare them with $\hat{R}$.

!!! warning "Multiple chains are experimental"
    Parallel MALA and multiple chains both get their speed from the same hardware.
    DEER does more total work than sequential MALA and only wins when that work runs in parallel on hardware that would otherwise sit idle.
    With many chains the hardware is already busy, so the extra work is no longer free, and plain sequential chains can end up faster.
    [Zoltowski et al. (2025)](https://arxiv.org/abs/2508.18413) observed this on a GPU: with 32 chains, sequential MALA was slightly faster than parallel MALA.
    How best to split resources between more chains and longer chains is still an open question.
    Run a few chains to compute diagnostics such as $\hat{R}$, not to sample faster.

Start Julia with several threads, for example `julia -t 4`, so that `MCMCThreads()` can run the chains at the same time:

```@example gs
chains = sample(
    model,
    sampler,
    MCMCThreads(),
    2_000,
    4;
    initial_params=fill(zeros(2), 4),
)

ess(chains)
```

```@example gs
rhat(chains)
```

`initial_params` now holds one starting point per chain.
Each chain runs on its own thread and still solves blocks of `T` transitions with DEER.
