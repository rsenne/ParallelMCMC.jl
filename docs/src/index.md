```@meta
CurrentModule = ParallelMCMC
```

# ParallelMCMC.jl

```@raw html
<p align="center">
  <img src="assets/logo.png" alt="ParallelMCMC logo" width="220">
</p>
```

ParallelMCMC.jl implements Markov chain Monte Carlo methods that parallelize
*across the sequence of samples*. Its main sampler, [`ParallelMALASampler`](@ref),
uses DEER to solve a block of MALA transitions together instead of waiting for
each transition before starting the next one.

This is different from running several independent chains at once. You can do
that too, but the package's main goal is to shorten the critical path *within*
each chain. Whether that produces a speedup depends on the target density,
trajectory length, AD backend, and hardware.

## Is this package a good fit?

| If you want to... | Start with... |
|---|---|
| Try the parallel-across-sequence algorithm | [`ParallelMALASampler`](@ref) |
| Tune a step size automatically | [`AdaptiveMALASampler`](@ref) |
| Check a model against ordinary sequential MALA | [`MALASampler`](@ref) |
| Sample a Turing model | [`DensityModel`](@ref) with the Turing extension |
| Run a large, batch-friendly target on a GPU | [GPU execution](15-gpu.md) |

Parallel MALA is most compelling when evaluating the target and its derivatives
is expensive enough to keep parallel hardware busy. For a small target on a
CPU, sequential MALA will often finish sooner.

## Install

ParallelMCMC supports Julia 1.10 and later.

```julia-repl
pkg> add ParallelMCMC
```

AD backends and GPU packages are optional. Add one only when your model needs
it; the first example below supplies its derivatives directly.

## A complete first run

The target here is a two-dimensional standard normal. Its gradient and
Hessian-vector product are simple enough to write down, so the example has no
optional dependencies.

```julia
using ParallelMCMC, FlexiChains, Random

logdensity(x) = -sum(abs2, x) / 2
gradient(x) = -x
hvp(x, v) = -v

model = DensityModel(
    logdensity,
    gradient,
    2;
    hvp=hvp,
    param_names=[:x1, :x2],
)

sampler = ParallelMALASampler(0.1; T=64)

chain = sample(
    MersenneTwister(42),
    model,
    sampler,
    1_000;
    initial_params=zeros(2),
    chain_type=VNChain,
)
```

`T=64` means that DEER solves blocks of 64 transitions. The call still returns
exactly 1,000 samples; `T` controls the internal block size, not the requested
chain length.

## Where to go next

- [Getting started](10-getting-started.md) walks through step-size tuning,
  running parallel MALA, reading the result, and common failure modes.
- [Defining models](12-models.md) covers hand-written derivatives, automatic
  differentiation, batched functions, parameter names, and Turing models.
- [GPU execution](15-gpu.md) explains when a GPU is worthwhile and shows a
  complete logistic-regression example.
- [How the algorithm works](20-algorithms.md) develops the DEER update and the
  diagonal parallel scan.
- [API reference](95-reference.md) lists constructors and low-level building
  blocks.

## Citation

The parallel-across-sequence method is described in:

> Zoltowski, D. M., Wu, S., Gonzalez, X., Kozachkov, L., & Linderman, S. W.
> (2025). *Parallelizing MCMC Across the Sequence Length*. NeurIPS 2025.
> [arXiv:2508.18413](https://arxiv.org/abs/2508.18413)

If you use ParallelMCMC.jl in published work, please use the package citation
in [`CITATION.cff`](https://github.com/rsenne/ParallelMCMC.jl/blob/main/CITATION.cff).
