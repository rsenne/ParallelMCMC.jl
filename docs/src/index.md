```@meta
CurrentModule = ParallelMCMC
```

# ParallelMCMC.jl

```@raw html
<p align="center">
  <img src="assets/logo.png" alt="ParallelMCMC logo" width="220">
</p>
```

**ParallelMCMC.jl** is a Julia package for **parallel-across-the-sequence** MCMC — algorithms that solve an entire trajectory of $T$ correlated steps *simultaneously* rather than one at a time.

```@raw html
<p align="center">
  <img src="assets/julia_deer_posterior.gif" alt="DEER trajectory estimates improving on a Julia-logo-shaped posterior" width="620">
</p>

<p align="center">
  <em>DEER iterates on a synthetic Julia-logo-shaped posterior: orange trajectory estimates move toward the taped MALA path over repeated trajectory solves.</em>
</p>
```

The flagship algorithm is **DEER** (Deterministic Equivalent-Expectation Recursion), which reformulates a chain of $T$ MALA steps as a fixed-point problem and solves it via Newton iterations, each costing $O(\log T)$ parallel work via an associative prefix scan.  The result is that wall-clock time per sample is *sublinear* in chain length on multi-core CPUs and GPUs.

The algorithm is described in:

> Zoltowski, D. M., Wu, S., Gonzalez, X., Kozachkov, L., & Linderman, S. W. (2025).
> **Parallelizing MCMC Across the Sequence Length.**
> *NeurIPS 2025.* [arXiv:2508.18413](https://arxiv.org/abs/2508.18413)

The included [`MALASampler`](@ref) and [`AdaptiveMALASampler`](@ref) are sequential MALA baselines — useful for correctness checks and step-size tuning, but not the primary focus of the package.

## Samplers

| Sampler | Role |
|---|---|
| [`ParallelMALASampler`](@ref) | **Primary** — parallel-across-sequence MALA via DEER; O(log T) per solve |
| [`MALASampler`](@ref) | Baseline — sequential MALA with a fixed step size |
| [`AdaptiveMALASampler`](@ref) | Baseline — sequential MALA with dual-averaging step-size adaptation |

All samplers implement the [AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl) interface and return [`FlexiChains`](https://pysm.dev/FlexiChains.jl) objects.

## Installation

To install the package into your current environment:

```julia-repl
pkg> add ParallelMCMC
```

## Quick start

### Define a model

The simplest entry point is [`DensityModel`](@ref), which wraps a log-density and its gradient:

```julia
using ParallelMCMC, FlexiChains
using ADTypes, Enzyme

# Example: 2-D standard normal
logp(x)      = -0.5 * sum(abs2, x)
grad_logp(x) = -x

model = DensityModel(logp, grad_logp, 2;
                     param_names=[:x1, :x2])
```

### DEER — parallel-across-sequence (primary algorithm)

```julia
sampler = ParallelMALASampler(0.1; T=64, jacobian=:stoch_diag,
                              backend=AutoEnzyme())

chain = sample(model, sampler, 500;
               chain_type=VNChain)
```

Each call to `sample` draws 500 samples by solving DEER trajectories of length `T=64` in parallel, re-solving from the last state when each trajectory is exhausted.

Specifying `chain_type=VNChain` returns a `FlexiChain{VarName}`, which has a parameter type of `VarName`.
This is intended for maximum ease of use; however, if you prefer parameter type of `Symbol` you can use `SymChain` instead.
See [the FlexiChains.jl docs](https://pysm.dev/FlexiChains.jl/) for more information about how to analyze and visualize chains.

### Sequential MALA baseline

```julia
sampler = AdaptiveMALASampler(0.1; n_warmup=500)

chain = sample(model, sampler, 2_000;
               chain_type=VNChain,
               discard_warmup=true,
               progress=true)
```

### Turing.jl integration

When `DynamicPPL` (part of Turing.jl) is loaded, a one-argument `DensityModel` constructor is available that wraps a `@model` directly.
Parameter names are automatically extracted, and values transformed back to the original model space:

```julia
using Turing, ParallelMCMC, FlexiChains

@model function normal_model(y)
    μ ~ Normal(0.0, 1.0)
    y ~ Normal(μ, 0.5)
end

model   = DensityModel(normal_model(1.5))
sampler = AdaptiveMALASampler(0.3; n_warmup=500)

chain = sample(model, sampler, 2_000;
               chain_type=VNChain,
               discard_warmup=true)
```

See [Getting Started](10-getting-started.md) for worked examples and guidance on choosing samplers, and [Algorithm Details](20-algorithms.md) for the mathematics behind DEER.

For Turing models the chain type used must be `VNChain` (not `SymChain`), as that is the natural parameter type for Turing models.

## Contributors

```@raw html
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rsenne"><img src="https://avatars.githubusercontent.com/u/50930199?v=4?s=100" width="100px;" alt="Ryan Senne"/><br /><sub><b>Ryan Senne</b></sub></a><br /><a href="#code-rsenne" title="Code">💻</a> <a href="#maintenance-rsenne" title="Maintenance">🚧</a> <a href="#test-rsenne" title="Tests">⚠️</a> <a href="#ideas-rsenne" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-rsenne" title="Reviewed Pull Requests">👀</a> <a href="#doc-rsenne" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/penelopeysm"><img src="https://avatars.githubusercontent.com/u/122629585?v=4?s=100" width="100px;" alt="Penelope Yong"/><br /><sub><b>Penelope Yong</b></sub></a><br /><a href="#code-penelopeysm" title="Code">💻</a> <a href="#test-penelopeysm" title="Tests">⚠️</a> <a href="#ideas-penelopeysm" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-penelopeysm" title="Reviewed Pull Requests">👀</a> <a href="#doc-penelopeysm" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://gdalle.github.io/"><img src="https://avatars.githubusercontent.com/u/22795598?v=4?s=100" width="100px;" alt="Guillaume Dalle"/><br /><sub><b>Guillaume Dalle</b></sub></a><br /><a href="#review-gdalle" title="Reviewed Pull Requests">👀</a> <a href="#ideas-gdalle" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://wsmoses.com"><img src="https://avatars.githubusercontent.com/u/1260124?v=4?s=100" width="100px;" alt="William Moses"/><br /><sub><b>William Moses</b></sub></a><br /><a href="#review-wsmoses" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
```
