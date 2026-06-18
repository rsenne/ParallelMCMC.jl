# GPU Execution

DEER's prefix-scan and per-step batch evaluations are array-type-agnostic, so the same `ParallelMALASampler` runs on either CPU or GPU.  This page covers when GPU pays off, the current limitations, and a worked example.

---

## When to use GPU

GPU pays off when the per-Newton-step work i.e., evaluating `logdensity_batch` and `grad_logdensity_batch` across the `T`-wide trajectory and the Hutchinson `probes` is large enough to saturate the device.

| Regime | Reach for |
|---|---|
| `D` ≲ a few hundred, lightweight `logp` | **CPU.** Per-step cost is dominated by overhead; GPU launch latency wins out. |
| Large `D`, or `logp` is itself a batched linear-algebra / kernel workload | **GPU.** `T` × `probes` Jacobian work fills the device. |
| Many independent chains via `MCMCThreads()` | **CPU**, scaled with threads.  GPU chains do not currently share a device pool. |

A useful heuristic: if your CPU `grad_logdensity_batch` already saturates BLAS at the batch sizes DEER passes it (`T` columns, plus `probes` HVP probes), GPU is likely to help.  If not, profile before switching.

---

## Limitations

### 1. DynamicPPL / Turing models are CPU-only

The one-argument `DensityModel(::DynamicPPL.Model)` constructor (and the `LogDensityProblems` path that backs it) executes the model with `Vector{Float64}` parameters on CPU.  There is no path today that runs a `@model` against `CuArray` parameters.

To run on GPU, hand-write `logdensity` / `grad_logdensity` (and the batched variants) directly over `CuArray`s and pass them to `DensityModel`.  This is shown in the example below.

### 2. Enzyme on GPU currently needs `pmcmc_matmul` / `pmcmc_dot` / `pmcmc_dotsum`

`ParallelMCMC` exports three thin wrappers:

```julia
pmcmc_matmul(A, B)  = A * B
pmcmc_dot(a, b)     = dot(a, b)
pmcmc_dotsum(A, B)  = sum(A .* B)
```

These exist purely so the package can register Enzyme rules against them without committing type piracy on `Base.*`, `Base.dot`, `Base.sum`.  The rules compute primal and cotangents with plain operators *outside* Enzyme's IR rewriter, which keeps them off the reverse-mode path that aborts on GPU.

```
unsupported tag gc-transition for
  call i32 @cuMemcpyDtoHAsync_v2(...) [ "jl_roots"(...), "gc-transition"() ]
UNREACHABLE executed at .../Enzyme/GradientUtils.cpp:309      (signal 6)
```

When you should use them:

- You are on GPU **and** using `AutoEnzyme()` as the AD backend.

When you do **not** need them:

- CPU code: plain `*`, `dot`, `sum` are faster and clearer.
- GPU code with `AutoMooncake(; config=nothing)`: Mooncake's own CUDA extension handles these operations natively.
- GPU code with `AutoZygote()`: Zygote uses ChainRules adjoints, which have cuBLAS-backed rules for `*` / `dot` / `sum` on `CuArray`.

These wrappers are a workaround pending upstream Enzyme support for GPU matmul rules.  Once that lands, plain `A * B` on `CuArray`s will work and the wrappers will be deprecated.

### 3. Some Enzyme GPU gradients hit a `gc-transition` abort. Staging them as single ops is a workaround

Certain `CuArray` gradient expressions abort during Enzyme compilation on GPU with the same `gc-transition` failure as limitation 2. In practice the fused form below aborts while the staged form compiles, so **writing the gradient as single-op stages is a reliable workaround**:

```julia
# ABORTS during Enzyme compile on GPU
gradlogp(β) = -β .- pmcmc_matmul(transpose(X), pmcmc_matmul(X, β)) ./ Float32(N)

# WORKS: built up one broadcast op at a time
function gradlogp(β)
    Xβ = pmcmc_matmul(X, β)
    Y  = pmcmc_matmul(transpose(X), Xβ)
    Y  = Y ./ Float32(N)
    Y  = Y .+ β
    return -Y
end
```

Mooncake does not have this restriction.

---

## Worked example: Bayesian logistic regression on GPU

A standard Bayesian logistic regression with a Gaussian prior; an example model where GPU acceleration actually pays off as `N` and `D` grow.  Posterior:

```math
\log p(\beta \mid y, X) = -\tfrac{1}{2}\|\beta\|^2 + \sum_{i=1}^{N} \bigl[ y_i \, x_i^\top \beta - \log(1 + e^{x_i^\top \beta}) \bigr],
```

with gradient $\nabla_\beta \log p = -\beta + X^\top (y - \sigma(X\beta))$, where $\sigma$ is the logistic sigmoid.

We use a numerically stable softplus on GPU:

```math
\log(1 + e^{z}) = \max(z, 0) + \log(1 + e^{-|z|}),
```

so neither the positive nor negative tail overflows in `Float32`.

### Synthetic data

```julia
using Random, CUDA

D = 50; N = 1000
rng    = MersenneTwister(0)
β_true = randn(rng, Float32, D)
X_cpu  = randn(rng, Float32, N, D)
probs  = 1f0 ./ (1f0 .+ exp.(-(X_cpu * β_true)))
y_cpu  = Float32.(rand(rng, N) .< probs)

X_gpu = CUDA.CuMatrix(X_cpu)
y_gpu = CUDA.CuVector(y_cpu)
```

### Mooncake backend (plain operators)

```julia
using ParallelMCMC, MCMCChains
using ADTypes, Mooncake

softplus(z) = log1p(exp(-abs(z))) + max(z, zero(z))

logp(β)     = -0.5f0 * sum(abs2, β) + dot(y_gpu, X_gpu * β) -
              sum(softplus.(X_gpu * β))
gradlogp(β) = let z = X_gpu * β
    -β .+ transpose(X_gpu) * (y_gpu .- 1f0 ./ (1f0 .+ exp.(-z)))
end

function logp_batch(B)
    Z     = X_gpu * B
    prior = -0.5f0 .* vec(sum(abs2, B; dims=1))
    ll    = vec(sum(y_gpu .* Z; dims=1)) .- vec(sum(softplus.(Z); dims=1))
    return prior .+ ll
end
function gradlogp_batch(B)
    Z = X_gpu * B
    P = 1f0 ./ (1f0 .+ exp.(-Z))
    return -B .+ transpose(X_gpu) * (y_gpu .- P)
end

model   = DensityModel(logp, gradlogp, D;
                       logdensity_batch=logp_batch,
                       grad_logdensity_batch=gradlogp_batch)
sampler = ParallelMALASampler(0.005f0;
                              T=16, damping=0.5f0,
                              backend=ADTypes.AutoMooncake(; config=nothing))

chain = sample(model, sampler, 1_600;
               initial_params=CUDA.zeros(Float32, D),
               chain_type=MCMCChains.Chains)
```

Posterior mean recovery error `‖β_post − β_true‖ / ‖β_true‖` should land in the 0.1–0.2 range after a few hundred post-warmup samples.

### Enzyme backend (requires `pmcmc_matmul)

Same model, with the GPU-Enzyme restrictions applied: every `*` becomes `pmcmc_matmul`, every `dot` becomes `pmcmc_dot`, and every gradient broadcast is expanded into single-op stages:

```julia
using ParallelMCMC, MCMCChains
using ADTypes, Enzyme

function logp(β)
    z      = pmcmc_matmul(X_gpu, β)
    az     = abs.(z)
    ez     = exp.(.-az)
    sp1    = log1p.(ez)
    sp2    = max.(z, 0f0)
    sp     = sp1 .+ sp2
    ll     = pmcmc_dot(y_gpu, z) - sum(sp)
    prior  = -0.5f0 * sum(abs2, β)
    return prior + ll
end

function gradlogp(β)
    z     = pmcmc_matmul(X_gpu, β)
    ez    = exp.(.-z)
    den   = 1f0 .+ ez
    p     = 1f0 ./ den
    resid = y_gpu .- p
    g_ll  = pmcmc_matmul(transpose(X_gpu), resid)
    g     = .-β
    g     = g .+ g_ll
    return g
end

function logp_batch(B)
    Z      = pmcmc_matmul(X_gpu, B)
    aZ     = abs.(Z)
    eZ     = exp.(.-aZ)
    sp1    = log1p.(eZ)
    sp2    = max.(Z, 0f0)
    SP     = sp1 .+ sp2
    yZ     = y_gpu .* Z
    ll     = vec(sum(yZ; dims=1)) .- vec(sum(SP; dims=1))
    prior  = -0.5f0 .* vec(sum(abs2, B; dims=1))
    return prior .+ ll
end

function gradlogp_batch(B)
    Z     = pmcmc_matmul(X_gpu, B)
    eZ    = exp.(.-Z)
    denZ  = 1f0 .+ eZ
    P     = 1f0 ./ denZ
    Resid = y_gpu .- P
    G_ll  = pmcmc_matmul(transpose(X_gpu), Resid)
    G     = .-B
    G     = G .+ G_ll
    return G
end

model   = DensityModel(logp, gradlogp, D;
                       logdensity_batch=logp_batch,
                       grad_logdensity_batch=gradlogp_batch)
sampler = ParallelMALASampler(0.005f0;
                              T=16, damping=0.5f0,
                              backend=ADTypes.AutoEnzyme())

chain = sample(model, sampler, 1_600;
               initial_params=CUDA.zeros(Float32, D),
               chain_type=MCMCChains.Chains)
```

The two snippets sample the same posterior; the difference is purely in what the AD backend can chew on.

---

## The AD-HVP fallback (and when to write your own)

DEER needs a Hessian–vector product $H v$ at every Newton step.  `DensityModel` accepts both `gradlogp` and (optionally) `hvp` / `hvp_batch`.  Depending on what you supply, the sampler does one of two things:

- **You supply `hvp` / `hvp_batch`.**  These run as plain kernels.  The AD backend is never invoked for HVPs.
- **You only supply `gradlogp` / `grad_logdensity_batch`.**  The sampler builds the HVP by differentiating your gradient — either a forward-mode pushforward of `gradlogp` ([`ForwardOnGrad`](https://github.com/rsenne/ParallelMCMC.jl/blob/main/src/DEER/DEER.jl), the default for most backends) or a reverse-mode gradient of `x -> dot(gradlogp(x), v)` ([`ReverseOnGrad`](https://github.com/rsenne/ParallelMCMC.jl/blob/main/src/DEER/DEER.jl), used for `AutoMooncake` and `AutoZygote`).  This is the **AD-HVP fallback**, and it is what the logistic-regression example above uses.

### When the fallback is the right call

- **Complex or composed models.**  Bayesian neural nets, hierarchical models with many transformations, mixtures, or anything where the Hessian has no convenient closed form.  Deriving and maintaining `hvp` by hand for these is error-prone; AD removes a whole class of bugs.
- **Prototyping.**  You want a correct sampler running before you optimize.  Drop in `gradlogp`, lean on AD, profile later.
- **Operations that AD libraries handle for free but are tedious to differentiate by hand.**  Special functions, link functions, log-sum-exp, normalizing constants of standard distributions, etc.
- **You want backend flexibility.**  With only `gradlogp` you can swap `AutoMooncake` ↔ `AutoEnzyme` ↔ `AutoZygote` to compare without rewriting model code.

### When to write your own analytical HVP

- **The HVP has a clean closed form.**  Quadratic priors, Gaussian likelihoods, GLMs (logistic, Poisson, probit) — the second derivative is a known function of intermediate quantities you already compute in `gradlogp`.  A few extra lines and you skip the AD pipeline entirely.
- **Performance matters and the AD compile is heavy.**  Enzyme and Mooncake both pay a one-shot compilation cost on the user's gradient.  For long-running chains this amortizes, but for many short runs the analytical HVP wins.
- **You're hitting AD-backend-specific GPU restrictions.**  The [Enzyme limitations](#2-enzyme-on-gpu-currently-needs-pmcmc_matmul-pmcmc_dot-pmcmc_dotsum) above (`pmcmc_*` wrappers, staged broadcasts) only matter when the AD backend is invoked.  Supplying analytical HVP sidesteps them — your `gradlogp` and `hvp` can use plain `*`, `dot`, `sum` even with `backend=AutoEnzyme()`, because the backend is held but never run.
- **You can reuse intermediates between gradient and HVP.**  When `hvp` shares $X\beta$, $\sigma(X\beta)$, or similar with the gradient computation, an analytical version can be both faster *and* shorter than what AD produces.

### Same example with analytical HVP

For Bayesian logistic regression with a Gaussian prior, the Hessian is

```math
\nabla^2 \log p = -I - X^\top \, \mathrm{diag}\bigl(\sigma(X\beta) \odot (1 - \sigma(X\beta))\bigr) \, X,
```

so

```math
H v = -v - X^\top \bigl[\sigma(X\beta) \odot (1 - \sigma(X\beta)) \odot (X v)\bigr].
```

Plug it in alongside `gradlogp` and the sampler stops invoking the AD path:

```julia
function hvp(β, v)
    z  = X_gpu * β
    σz = 1f0 ./ (1f0 .+ exp.(-z))
    w  = σz .* (1f0 .- σz)
    -v .- transpose(X_gpu) * (w .* (X_gpu * v))
end
function hvp_batch(B, V)
    Z = X_gpu * B
    Σ = 1f0 ./ (1f0 .+ exp.(-Z))
    W = Σ .* (1f0 .- Σ)
    -V .- transpose(X_gpu) * (W .* (X_gpu * V))
end

model = DensityModel(logp, gradlogp, D;
                     logdensity_batch=logp_batch,
                     grad_logdensity_batch=gradlogp_batch,
                     hvp=hvp, hvp_batch=hvp_batch)

# `backend` is still required by the API, but is never invoked.
sampler = ParallelMALASampler(0.005f0;
                              T=16, damping=0.5f0,
                              backend=ADTypes.AutoMooncake(; config=nothing))
```

This recovers the same posterior as the fallback version — the only difference is that HVPs come from a plain matmul instead of a reverse-mode pass over `gradlogp`.

---

## Picking an AD backend on GPU

| Backend | GPU support | Restrictions | Recommended when |
|---|---|---|---|
| `AutoEnzyme()` | Via `pmcmc_matmul` / `pmcmc_dot` / `pmcmc_dotsum` and single-op-broadcast gradients | The restrictions above | You want Enzyme's reverse-mode performance and are willing to follow the rules above |
| `AutoMooncake(; config=nothing)` | Native (no wrappers) | Slower compile | You want plain Julia operators with no rewriting |
| `AutoZygote()` | Native (no wrappers) | ChainRules-based; reverse-only, plus no in-place mutation in `gradlogp`| You want plain Julia operators and are already on the ChainRules ecosystem |
| `AutoForwardDiff()` | Native; works with `*` / `dot` / `sum` / broadcasts on `CuArray{Dual}`| Cost scales as O(D)| Low-D models, prototyping, validating Mooncake/Enzyme results |

If you supply `hvp` / `hvp_batch` analytically, none of the above matters — those run as plain kernels and the sampler does not invoke a backend.
