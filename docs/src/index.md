```@meta
CurrentModule = ParallelMCMC
```

# ParallelMCMC.jl

**ParallelMCMC.jl** is a Julia package for **parallel-across-the-sequence** MCMC — algorithms that solve an entire trajectory of $T$ correlated steps *simultaneously* rather than one at a time.

The flagship algorithm is **DEER** (Deterministic Equivalent-Expectation Recursion), which reformulates a chain of $T$ MALA steps as a fixed-point problem and solves it via Newton iterations, each costing $O(\log T)$ parallel work via an associative prefix scan.  The result is that wall-clock time per sample is *sublinear* in chain length on multi-core CPUs and GPUs.

The algorithm is described in:

> Zoltowski, D., Wu, Y., Gonzalez, D., Kozachkov, L., & Linderman, S. (2025).
> **Parallelizing MCMC Across the Sequence Length.**
> *NeurIPS 2025.* [arXiv:2508.18413](https://arxiv.org/abs/2508.18413)

The included [`MALASampler`](@ref) and [`AdaptiveMALASampler`](@ref) are sequential MALA baselines — useful for correctness checks and step-size tuning, but not the primary focus of the package.

## Samplers

| Sampler | Role |
|---|---|
| [`DEERSampler`](@ref) | **Primary** — parallel-across-sequence MALA via DEER; O(log T) per solve |
| [`MALASampler`](@ref) | Baseline — sequential MALA with a fixed step size |
| [`AdaptiveMALASampler`](@ref) | Baseline — sequential MALA with dual-averaging step-size adaptation |

All samplers implement the [AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl) interface and return [`MCMCChains.Chains`](https://github.com/TuringLang/MCMCChains.jl) objects.

## Installation

ParallelMCMC.jl is a registered Julia package:

```julia-repl
julia> ] add ParallelMCMC
```

## Quick start

### Define a model

The simplest entry point is [`DensityModel`](@ref), which wraps a log-density and its gradient:

```julia
using ParallelMCMC, MCMCChains

# Example: 2-D standard normal
logp(x)      = -0.5 * sum(abs2, x)
grad_logp(x) = -x

model = DensityModel(logp, grad_logp, 2;
                     param_names=[:x1, :x2])
```

### DEER — parallel-across-sequence (primary algorithm)

```julia
sampler = DEERSampler(0.1; T=64, jacobian=:diag)

chain = sample(model, sampler, 500;
               chain_type=MCMCChains.Chains)
```

Each call to `sample` draws 500 samples by solving DEER trajectories of length `T=64` in parallel, re-solving from the last state when each trajectory is exhausted.

### Sequential MALA baseline

```julia
sampler = AdaptiveMALASampler(0.1; n_warmup=500)

chain = sample(model, sampler, 2_000;
               chain_type=MCMCChains.Chains,
               discard_warmup=true,
               progress=true)
```

### Turing.jl integration

When `DynamicPPL` (part of Turing.jl) is loaded, a one-argument `DensityModel` constructor is available that wraps a `@model` directly, extracting parameter names automatically:

```julia
using Turing, ParallelMCMC, MCMCChains

@model function normal_model(y)
    μ ~ Normal(0.0, 1.0)
    y ~ Normal(μ, 0.5)
end

model   = DensityModel(normal_model(1.5))
sampler = AdaptiveMALASampler(0.3; n_warmup=500)

chain = sample(model, sampler, 2_000;
               chain_type=MCMCChains.Chains,
               discard_warmup=true)
```

See [Getting Started](10-getting-started.md) for worked examples and guidance on choosing samplers, and [Algorithm Details](20-algorithms.md) for the mathematics behind DEER.

## Contributors

```@raw html
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
```
