```@meta
CurrentModule = ParallelMCMC
```

# ParallelMCMC.jl

```@raw html
<p align="center">
  <img src="assets/logo.png" alt="ParallelMCMC logo" width="220">
</p>
```

ParallelMCMC.jl parallelizes Markov chain Monte Carlo (MCMC) *within* a single chain.
Its main sampler, [`ParallelMALASampler`](@ref), draws the random numbers for a block of MALA transitions in advance.
It then solves for all states in the block at once with DEER, a Newton method whose linear solves are parallel scans.
You can still run several chains in parallel on top of this.

Parallel MALA does more total work than sequential MALA, so it finishes sooner only when that work can be spread over many cores or a GPU.
The speedup depends on the target, the block length `T`, the AD backend, and the hardware.

```@raw html
<p align="center">
  <img src="assets/julia_deer_posterior.gif" alt="DEER trajectory estimates improving on a Julia-logo-shaped posterior" width="620">
</p>
<p align="center">
  <em>DEER iterates on a synthetic Julia-logo-shaped posterior: orange trajectory estimates move toward the taped MALA path over repeated trajectory solves.</em>
</p>
```

## Which sampler to use

| If you want to... | Start with... |
|---|---|
| Try the parallel-across-sequence algorithm | [`ParallelMALASampler`](@ref) |
| Tune a step size automatically | [`AdaptiveMALASampler`](@ref) |
| Check a model against ordinary sequential MALA | [`MALASampler`](@ref) |
| Run a large, batch-friendly target on a GPU | [GPU execution](15-gpu.md) |

## Install

ParallelMCMC supports Julia 1.10 and later.

```julia-repl
pkg> add ParallelMCMC
```

ParallelMCMC does not depend on any AD or GPU package, so you load the ones your model needs.
The example below writes its derivatives by hand and needs only ParallelMCMC.

## A complete first run

The target is a two-dimensional standard normal, $\log p(x) = -\tfrac{1}{2}\|x\|^2$.
Its gradient is $-x$ and its Hessian-vector product is $-v$.

```@example index
using ParallelMCMC, Random

logdensity(x) = -sum(abs2, x) / 2
gradient(x) = -x
hvp(x, v) = -v

model = DensityModel(logdensity, gradient, 2; hvp=hvp, param_names=[:x1, :x2])
sampler = ParallelMALASampler(0.1; T=64)
chain = sample(MersenneTwister(42), model, sampler, 1_000; initial_params=zeros(2))
```

`T=64` makes DEER solve 64 transitions at a time.
It does not change the length of the chain: `sample` still returns 1,000 samples.

## Where to go next

- [Getting started](10-getting-started.md) walks through step-size tuning, running parallel MALA, and reading the result.
- [Defining models](12-models.md) covers hand-written derivatives, automatic differentiation, batched functions, parameter names, and LogDensityProblems models.
- [GPU execution](15-gpu.md) explains when a GPU is worthwhile and shows a complete logistic-regression example.
- [Troubleshooting](16-troubleshooting.md) covers common errors and failure modes.
- [How the algorithm works](20-algorithms.md) develops the DEER update and the diagonal parallel scan.
- [API reference](95-reference.md) lists constructors and low-level building blocks.

## Citation

The parallel-across-sequence method is described in:

> Zoltowski, D. M., Wu, S., Gonzalez, X., Kozachkov, L., & Linderman, S. W.
> (2025). *Parallelizing MCMC Across the Sequence Length*. NeurIPS 2025.
> [arXiv:2508.18413](https://arxiv.org/abs/2508.18413)

If you use ParallelMCMC.jl in published work, please use the package citation in [`CITATION.cff`](https://github.com/rsenne/ParallelMCMC.jl/blob/main/CITATION.cff).
