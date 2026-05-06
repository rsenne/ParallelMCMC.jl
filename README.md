# ParallelMCMC

<p align="center">
  <img src="docs/src/assets/logo.png" alt="ParallelMCMC logo" width="220">
</p>

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://rsenne.github.io/ParallelMCMC.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://rsenne.github.io/ParallelMCMC.jl/dev)
[![Test workflow status](https://github.com/rsenne/ParallelMCMC.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/rsenne/ParallelMCMC.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/rsenne/ParallelMCMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rsenne/ParallelMCMC.jl)
[![Docs workflow Status](https://github.com/rsenne/ParallelMCMC.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/rsenne/ParallelMCMC.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/rsenne/ParallelMCMC.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

<p align="center">
  <img src="docs/src/assets/julia_deer_posterior.gif" alt="DEER trajectory estimates improving on a Julia-logo-shaped posterior" width="620">
</p>

<p align="center">
  <em>DEER iterates on a synthetic Julia-logo-shaped posterior: orange trajectory estimates move toward the taped MALA path over repeated trajectory solves.</em>
</p>

## What this package does

**ParallelMCMC.jl** implements *parallel-across-the-sequence* MCMC in Julia: instead of generating samples one at a time, an entire trajectory of $T$ correlated steps is solved *simultaneously*. This makes wall-clock time per sample sublinear in chain length on multi-core CPUs and GPUs, where conventional sequential MCMC scales linearly.

The flagship algorithm is **DEER** (Lim et al. 2024; Gonzalez et al. 2024), which reformulates a chain of $T$ MALA steps as a fixed-point problem and solves it with Newton iterations. Each iteration linearizes the per-step transition around the current trajectory guess and resolves the resulting linear recursion in $O(\log T)$ parallel work via an associative prefix scan. With shared input randomness, DEER converges to the exact sequential MALA trace up to a numerical tolerance — typically in tens of iterations even for chains of tens of thousands of samples.

The approach and its scaling tricks (stochastic Hutchinson Jacobian estimators, damping, sliding windows) are described in:

> Zoltowski, D. M., Wu, S., Gonzalez, X., Kozachkov, L., & Linderman, S. W. (2025).
> **Parallelizing MCMC Across the Sequence Length.** *NeurIPS 2025.*
> [arXiv:2508.18413](https://arxiv.org/abs/2508.18413)

### Samplers

| Sampler | Role |
|---|---|
| [`ParallelMALASampler`](src/interface.jl) | **Primary** — parallel-across-sequence MALA via DEER; $O(\log T)$ per solve |
| [`MALASampler`](src/interface.jl) | Baseline — sequential MALA with a fixed step size |
| [`AdaptiveMALASampler`](src/interface.jl) | Baseline — sequential MALA with dual-averaging step-size adaptation |

All samplers implement the [AbstractMCMC](https://github.com/TuringLang/AbstractMCMC.jl) interface and return [`MCMCChains.Chains`](https://github.com/TuringLang/MCMCChains.jl) objects, so they slot into existing Turing.jl / AbstractMCMC workflows.


### Quick start

ParallelMCMC can be installed via Julia's package manager. In the Julia REPL, press `]` to enter pkg mode, then run:

```julia-repl
pkg> add ParallelMCMC
```

```julia
using ParallelMCMC, MCMCChains

logp(x)      = -0.5 * sum(abs2, x)            # 2-D standard normal
grad_logp(x) = -x

model   = DensityModel(logp, grad_logp, 2; param_names=[:x1, :x2])
sampler = ParallelMALASampler(0.1; T=64, jacobian=:stoch_diag)

chain = sample(model, sampler, 500; chain_type=MCMCChains.Chains)
```

See the [Getting Started guide](docs/src/10-getting-started.md) for worked examples, GPU usage, Turing.jl integration, and step-size tuning.

## How to Cite

If you use ParallelMCMC.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/rsenne/ParallelMCMC.jl/blob/main/CITATION.cff).

## Contributing

If you want to contribute, start with the [contributing guide on GitHub](docs/src/90-contributing.md) or the [documentation site](https://rsenne.github.io/ParallelMCMC.jl/dev/90-contributing/).

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
