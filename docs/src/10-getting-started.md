# Getting Started

ParallelMCMC.jl is built around **parallel-across-the-sequence** MCMC.  The core algorithm is [`ParallelMALASampler`](@ref), which solves an entire trajectory of correlated steps simultaneously.  [`MALASampler`](@ref) and [`AdaptiveMALASampler`](@ref) are sequential MALA baselines, included mainly for correctness testing and step-size exploration before running DEER.

## Defining a model

All samplers take a [`DensityModel`](@ref) as their first argument.  A `DensityModel` wraps:

- `logdensity(x) -> Real` — log-density of the target (up to a constant)
- `grad_logdensity(x) -> AbstractVector` — its gradient
- optional `hvp`, `logdensity_batch`, `grad_logdensity_batch`, and `hvp_batch` helpers for faster DEER Jacobian updates
- `dim::Int` — dimension of the parameter space

```julia
using ParallelMCMC, FlexiChains
using ADTypes, Enzyme

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

## ParallelMALASampler

[`ParallelMALASampler`](@ref) reformulates a trajectory of `T` MALA steps as a fixed-point problem and solves it via Newton iterations, each of which costs $O(\log T)$ parallel work via an associative prefix scan.  Wall-clock time per sample is therefore sublinear in chain length on multi-core CPUs and GPUs.

```julia
sampler = ParallelMALASampler(0.1; T=64, jacobian=:stoch_diag, damping=0.5,
                              backend=AutoEnzyme())

chain = sample(model, sampler, 500;
               chain_type=VNChain, progress=true)
```

`sample` requests 500 total samples.  Internally, DEER solves trajectories of length `T=64` and returns each column of the solved trajectory as a separate sample.  When the trajectory is exhausted a new noise tape is drawn and DEER re-solves from the last state.

### Trajectory length `T`

A larger `T` means more samples per DEER solve and better parallelism.  On GPU or with many CPU threads the cost per sample decreases roughly as $1/T$ (up to the $O(\log T)$ scan overhead).  A typical starting point is `T=64`; increase to `T=128` or `T=256` for long sampling runs on parallel hardware.

### Jacobian modes

The `jacobian` keyword controls how the per-step Jacobian is approximated during each Newton iteration:

| Mode | Cost per step | Notes |
|---|---|---|
| `:stoch_diag` (default) | `probes` JVPs | Hutchinson estimator; scalable default for high dimensions |
| `:diag` | `D` JVPs | Exact diagonal; useful for low-dimensional checks |

For high-dimensional targets, `:stoch_diag` with a small number of `probes` is a good trade-off:

```julia
sampler = ParallelMALASampler(0.1; T=128, jacobian=:stoch_diag, probes=2,
                              backend=AutoEnzyme())
```

### Damping

Setting `damping < 1` blends the Newton update with the previous iterate, which improves convergence on poorly conditioned targets.  The default is `0.5`; values in `[0.3, 0.7]` are typical.

### GPU execution

The prefix-scan kernel runs as pure array broadcasts and is array-type-agnostic.  To run DEER on GPU:

1. Implement `logdensity` and `grad_logdensity` using GPU-compatible operations.
2. Pass for example, `backend = ADTypes.AutoEnzyme()` to `ParallelMALASampler`.
3. Pass a `CuVector` as `initial_params` to `sample`.

GPU execution has its own caveats (Turing models are CPU-only, `AutoEnzyme()` currently needs the `pmcmc_matmul` / `pmcmc_dot` / `pmcmc_dotsum` wrappers, gradients should avoid scalar indexing).  See [GPU Execution](15-gpu.md) for the worked example and the full set of limitations.

---

## Turing.jl integration

ParallelMCMC.jl integrates with Turing.jl models through the `DynamicPPL` and `LogDensityProblems` extensions.

### One-step convenience constructor

Load `DynamicPPL` (part of Turing.jl) and a single-argument `DensityModel` constructor becomes available:

```julia
using Turing, ParallelMCMC, FlexiChains

@model function normal_model(y)
    μ ~ Normal(0.0, 1.0)
    y ~ Normal(μ, 0.5)
end

model = DensityModel(normal_model(1.5))   # param_names=[:μ] extracted automatically

chain = sample(model, ParallelMALASampler(0.1; T=64, backend=AutoEnzyme()), 500;
               chain_type=VNChain)
```

Much like Turing's own samplers, the resulting chain will always have parameters in the original (possibly constrained) space, even though the MCMC sampling itself is performed in unconstrained space.
Furthermore, parameter names are automatically extracted from the Turing model (and will always be the same as those when using Turing's own samplers).

Note that when sampling with a Turing model the returned chain will have `:logjoint`, `:logprior`, and `:loglikelihood` columns, since Turing models provide enough information to separate these contributions to the log-density.
This is in contrast to sampling with a manually constructed `DensityModel`, which returns only a single `:logp` column.

### Manual `LogDensityProblems` path

For explicit control over the AD backend, construct the `LogDensityFunction`
directly with DynamicPPL's `adtype` interface:

```julia
using Turing, LogDensityProblems, ADTypes
using ParallelMCMC, FlexiChains

ld = DynamicPPL.LogDensityFunction(
    normal_model(1.5),
    DynamicPPL.getlogjoint_internal,
    DynamicPPL.LinkAll();
    adtype=ADTypes.AutoEnzyme(),
)

model = DensityModel(ld; param_names=[:μ])
```

This also accepts any other `LogDensityProblems`-compatible object.
As above, the returned chain will always contain parameters in the original space: the use of `LinkAll()` above only stipulates that MCMC sampling itself is to be performed in unconstrained space, and has no result on the form of the returned chain.

---

## Parallel chains

All samplers support `MCMCThreads()`.  Start Julia with multiple threads (e.g. `julia -t 4`):

```julia
chain = sample(model, ParallelMALASampler(0.1; T=64, backend=AutoEnzyme()),
               MCMCThreads(), 500, 4;
               chain_type=VNChain)
```

`FlexiChains` computes R-hat and ESS across chains automatically:

```julia
ess(chain)
```

To calculate intra-chain metrics, you can pass `dims=:iter` [as described in the FlexiChains docs](https://pysm.dev/FlexiChains.jl/stable/summarising/#Individual-statistics):

```julia
ess(chain; dims=:iter)
```

---

## [Specifying parameter names](@id parameter-names)

For manually constructed `DensityModel`s, you can optionally specify parameter names with the `param_names` keyword.
The resulting `FlexiChain` object will then have named entries for each parameter.

If you do not specify `param_names`, the chain will store a single vector-valued parameter called `x` of length `D`.

You can specify parameter names as a collection of either:

- `Symbol`s (e.g. `[:x1, :x2]`);
- `VarName`s (e.g. `[@varname(x[1]), @varname(x[2])]`) if `chain_type=VNChain`; or
- A tuple of the above, *plus* a size. In this case, a total of `prod(size)` entries will be allocated to the named parameter, and the results in the chain will be reshaped to that size.

The above can be mixed and matched as desired, as long as the total number of parameters matches the dimension of the model.
For example:

```julia
# Three scalar parameters, called `x1` through `x3`
model = DensityModel(...; param_names=[:x1, :x2, :x3])

# One scalar parameter called `x`, and a 1x2 matrix parameter called `y`
model = DensityModel(...; param_names=[:x, (:y, (1, 2))])
```

For Turing.jl models, parameter names are automatically derived from the model and do not need to be specified manually.

---

## Sequential MALA baselines

[`MALASampler`](@ref) and [`AdaptiveMALASampler`](@ref) are standard sequential MALA implementations.  They are useful for:

- Verifying that a model and step size are set up correctly before running DEER
- Obtaining a reference chain to compare against
- Step-size exploration (run `AdaptiveMALASampler` first, read off the frozen `ε̄`, pass it to `ParallelMALASampler`)

```julia
# Step 1: find a good step size with adaptive MALA
baseline = sample(model, AdaptiveMALASampler(0.1; n_warmup=500), 600;
                  chain_type=VNChain, discard_warmup=true)

# Read off the frozen step size from the last internal value
eps_tuned = baseline[end, :step_size, 1]

# Step 2: run DEER with the tuned step size
chain = sample(model, ParallelMALASampler(eps_tuned; T=64, backend=AutoEnzyme()), 2_000;
               chain_type=VNChain)
```

See the [`MALASampler`](@ref) and [`AdaptiveMALASampler`](@ref) reference pages for the full keyword listing.
