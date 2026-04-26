# Getting Started

ParallelMCMC.jl is built around **parallel-across-the-sequence** MCMC.  The core algorithm is [`ParallelMALASampler`](@ref), which solves an entire trajectory of correlated steps simultaneously.  [`MALASampler`](@ref) and [`AdaptiveMALASampler`](@ref) are sequential MALA baselines, included mainly for correctness testing and step-size exploration before running DEER.

## Defining a model

All samplers take a [`DensityModel`](@ref) as their first argument.  A `DensityModel` wraps:

- `logdensity(x) -> Real` — log-density of the target (up to a constant)
- `grad_logdensity(x) -> AbstractVector` — its gradient
- optional `hvp`, `logdensity_batch`, `grad_logdensity_batch`, and `hvp_batch` helpers for faster DEER Jacobian updates
- `dim::Int` — dimension of the parameter space

```julia
using ParallelMCMC, MCMCChains

# Banana-shaped target in 2-D
function logp(x)
    -0.5 * (x[1]^2 / 100 + (x[2] + 0.1*x[1]^2 - 1)^2)
end

function grad_logp(x)
    dx1 = -x[1]/100 - 0.2*x[1]*(x[2] + 0.1*x[1]^2 - 1)
    dx2 = -(x[2] + 0.1*x[1]^2 - 1)
    [dx1, dx2]
end

model = DensityModel(logp, grad_logp, 2; param_names=[:x1, :x2])
```

---

## ParallelMALASampler — the primary algorithm

[`ParallelMALASampler`](@ref) reformulates a trajectory of `T` MALA steps as a fixed-point problem and solves it via Newton iterations, each of which costs $O(\log T)$ parallel work via an associative prefix scan.  Wall-clock time per sample is therefore sublinear in chain length on multi-core CPUs and GPUs.

```julia
sampler = ParallelMALASampler(0.1; T=64, jacobian=:diag, damping=0.5)

chain = sample(model, sampler, 500;
               chain_type=MCMCChains.Chains, progress=true)
```

`sample` requests 500 total samples.  Internally, DEER solves trajectories of length `T=64` and returns each column of the solved trajectory as a separate sample.  When the trajectory is exhausted a new noise tape is drawn and DEER re-solves from the last state.

### Trajectory length `T`

A larger `T` means more samples per DEER solve and better parallelism.  On GPU or with many CPU threads the cost per sample decreases roughly as $1/T$ (up to the $O(\log T)$ scan overhead).  A typical starting point is `T=64`; increase to `T=128` or `T=256` for long sampling runs on parallel hardware.

### Jacobian modes

The `jacobian` keyword controls how the per-step Jacobian is approximated during each Newton iteration:

| Mode | Cost per step | Notes |
|---|---|---|
| `:diag` (default) | 1 full Jacobian | Exact diagonal; good default |
| `:stoch_diag` | `probes` JVPs | Hutchinson estimator; better in high dimensions |
| `:full` | 1 full Jacobian | Full matrix step; rarely needed |

For high-dimensional targets, `:stoch_diag` with a small number of `probes` is a good trade-off:

```julia
sampler = ParallelMALASampler(0.1; T=128, jacobian=:stoch_diag, probes=2)
```

### Damping

Setting `damping < 1` blends the Newton update with the previous iterate, which improves convergence on poorly conditioned targets.  The default is `0.5`; values in `[0.3, 0.7]` are typical.

### GPU execution

The prefix-scan kernel runs as pure array broadcasts and is array-type-agnostic.  To run DEER on GPU:

1. Implement `logdensity` and `grad_logdensity` using GPU-compatible operations.
2. Pass `backend = ADTypes.AutoEnzyme()` to `ParallelMALASampler`.
3. Pass a `CuVector` as `initial_params` to `sample`.

```julia
using CUDA, ADTypes

sampler = ParallelMALASampler(0.1f0; T=64, backend=ADTypes.AutoEnzyme())

chain = sample(model, sampler, 500;
               initial_params=CUDA.randn(Float32, 2),
               chain_type=MCMCChains.Chains)
```

---

## Turing.jl integration

ParallelMCMC.jl integrates with Turing.jl models through the `LogDensityProblems` extension.

### One-step convenience constructor

Load `DynamicPPL` (part of Turing.jl) and a single-argument `DensityModel` constructor becomes available.  It extracts parameter names automatically:

```julia
using Turing, ParallelMCMC, MCMCChains

@model function normal_model(y)
    μ ~ Normal(0.0, 1.0)
    y ~ Normal(μ, 0.5)
end

model = DensityModel(normal_model(1.5))   # param_names=[:μ] extracted automatically

chain = sample(model, ParallelMALASampler(0.1; T=64), 500;
               chain_type=MCMCChains.Chains)
```

### Manual `LogDensityProblems` path

For explicit control over the AD backend, wrap the model yourself:

```julia
using Turing, LogDensityProblems, LogDensityProblemsAD, ADTypes
using ParallelMCMC, MCMCChains

ld  = DynamicPPL.LogDensityFunction(normal_model(1.5))
ldg = LogDensityProblemsAD.ADgradient(ADTypes.AutoMooncake(; config=nothing), ld)

model = DensityModel(ldg; param_names=[:μ])
```

This also accepts any other `LogDensityProblems`-compatible object.

---

## Parallel chains

All samplers support `MCMCThreads()`.  Start Julia with multiple threads (e.g. `julia -t 4`):

```julia
chain = sample(model, ParallelMALASampler(0.1; T=64), MCMCThreads(), 500, 4;
               chain_type=MCMCChains.Chains)
```

`MCMCChains` computes R-hat and ESS across chains automatically.

---

## Sequential MALA baselines

[`MALASampler`](@ref) and [`AdaptiveMALASampler`](@ref) are standard sequential MALA implementations.  They are useful for:

- Verifying that a model and step size are set up correctly before running DEER
- Obtaining a reference chain to compare against
- Step-size exploration (run `AdaptiveMALASampler` first, read off the frozen `ε̄`, pass it to `ParallelMALASampler`)

```julia
# Step 1: find a good step size with adaptive MALA
baseline = sample(model, AdaptiveMALASampler(0.1; n_warmup=500), 600;
                  chain_type=MCMCChains.Chains, discard_warmup=true)

# Read off the frozen step size from the last internal value
eps_tuned = baseline[end, :step_size, 1]

# Step 2: run DEER with the tuned step size
chain = sample(model, ParallelMALASampler(eps_tuned; T=64), 2_000;
               chain_type=MCMCChains.Chains)
```

See the [`MALASampler`](@ref) and [`AdaptiveMALASampler`](@ref) reference pages for the full keyword listing.
