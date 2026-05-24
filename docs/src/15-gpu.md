# GPU Execution

DEER's prefix-scan and per-step batch evaluations are array-type-agnostic, so the same `ParallelMALASampler` runs on either CPU or GPU.  This page covers when GPU pays off, the current limitations, and a worked example.

---

## When to use GPU

GPU pays off when the per-Newton-step work — evaluating `logdensity_batch` and `grad_logdensity_batch` across the `T`-wide trajectory and the Hutchinson `probes` — is large enough to saturate the device.  The prefix scan itself is `O(\log T)` on either backend; you are not buying scan speedup by moving to GPU, you are buying batched gradient evaluation speedup.

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

These exist purely so the package can register Enzyme rules against them without committing type piracy on `Base.*`, `Base.dot`, `Base.sum`.  The rules compute primal and cotangents with plain operators *outside* Enzyme's IR rewriter, which sidesteps the `unsupported tag gc-transition` abort Enzyme hits when it tries to lower cuBLAS bundles or the `cuMemcpyDtoHAsync_v2` emitted by CuArray scalar reductions.

When you should use them:

- You are on GPU **and** using `AutoEnzyme()` as the AD backend.

When you do **not** need them:

- CPU code — plain `*`, `dot`, `sum` are faster and clearer.
- GPU code with `AutoMooncake(; config=nothing)` — Mooncake's own CUDA extension handles these operations natively.
- GPU code with `AutoZygote()` — Zygote uses ChainRules adjoints, which have cuBLAS-backed rules for `*` / `dot` / `sum` on `CuArray`.

These wrappers are a workaround pending upstream Enzyme support for GPU matmul rules.  Once that lands, plain `A * B` on `CuArray`s will work and the wrappers will be deprecated.

### 3. Gradients must be written as single-op broadcasts, not fused expressions

Enzyme on GPU can lower a single broadcast kernel at a time but aborts when a kernel fuses multiple operations.  This means the fused form

```julia
# BROKEN: aborts during Enzyme compile on GPU
gradlogp(β) = -β .- pmcmc_matmul(transpose(X), pmcmc_matmul(X, β)) ./ Float32(N)
```

must be written as a sequence of single-op stages:

```julia
# WORKS: each line is one CUDA kernel Enzyme can differentiate
function gradlogp(β)
    Xβ = pmcmc_matmul(X, β)
    Y  = pmcmc_matmul(transpose(X), Xβ)
    Y  = Y ./ Float32(N)
    Y  = Y .+ β
    return -Y
end
```

Each intermediate assignment forces Julia to emit a separate broadcast kernel with a small enough gc-transition bundle that Enzyme can handle.  This restriction goes away when the upstream Enzyme fix lands; until then it applies to every gradient written for the GPU + Enzyme path.

Mooncake does not have this restriction.

---

## Worked example: GPU multivariate Gaussian

Target: $\log p(\beta) = -\tfrac{1}{2}(\|\beta\|^2 + \|X\beta\|^2 / N)$, with $X \in \mathbb{R}^{N \times D}$ on GPU.

### Enzyme backend (requires `pmcmc_matmul`)

```julia
using ParallelMCMC, MCMCChains
using ADTypes, Enzyme
using CUDA, Random

D = 20
N = 64

rng   = MersenneTwister(20251231)
X_gpu = CUDA.CuMatrix(randn(rng, Float32, N, D))

function logp(β)
    Xβ = pmcmc_matmul(X_gpu, β)
    -0.5f0 * (sum(abs2, β) + sum(abs2, Xβ) / Float32(N))
end

function gradlogp(β)
    Xβ = pmcmc_matmul(X_gpu, β)
    Y  = pmcmc_matmul(transpose(X_gpu), Xβ)
    Y  = Y ./ Float32(N)
    Y  = Y .+ β
    return -Y
end

function logp_batch(B)
    XB = pmcmc_matmul(X_gpu, B)
    -0.5f0 .* (vec(sum(abs2, B; dims=1)) .+ vec(sum(abs2, XB; dims=1)) ./ Float32(N))
end

function gradlogp_batch(B)
    XB = pmcmc_matmul(X_gpu, B)
    Y  = pmcmc_matmul(transpose(X_gpu), XB)
    Y  = Y ./ Float32(N)
    Y  = Y .+ B
    return -Y
end

model = DensityModel(logp, gradlogp, D;
                     logdensity_batch=logp_batch,
                     grad_logdensity_batch=gradlogp_batch)

sampler = ParallelMALASampler(0.05f0;
                              T=16, damping=0.5f0,
                              backend=ADTypes.AutoEnzyme())

chain = sample(model, sampler, 1_600;
               initial_params=CUDA.zeros(Float32, D),
               chain_type=MCMCChains.Chains)
```

### Mooncake backend (plain operators)

The same model with `AutoMooncake()`.  Note the wrappers are gone — plain `*`, `transpose`, and `sum` are differentiated by Mooncake's CUDA extension directly:

```julia
using ParallelMCMC, MCMCChains
using ADTypes, Mooncake
using CUDA, Random

D = 20
N = 64

rng   = MersenneTwister(20251231)
X_gpu = CUDA.CuMatrix(randn(rng, Float32, N, D))

logp(β)     = -0.5f0 * (sum(abs2, β) + sum(abs2, X_gpu * β) / Float32(N))
gradlogp(β) = -(β .+ (transpose(X_gpu) * (X_gpu * β)) ./ Float32(N))

logp_batch(B)     = -0.5f0 .* (vec(sum(abs2, B; dims=1))
                                .+ vec(sum(abs2, X_gpu * B; dims=1)) ./ Float32(N))
gradlogp_batch(B) = -(B .+ (transpose(X_gpu) * (X_gpu * B)) ./ Float32(N))

model = DensityModel(logp, gradlogp, D;
                     logdensity_batch=logp_batch,
                     grad_logdensity_batch=gradlogp_batch)

sampler = ParallelMALASampler(0.05f0;
                              T=16, damping=0.5f0,
                              backend=ADTypes.AutoMooncake(; config=nothing))

chain = sample(model, sampler, 1_600;
               initial_params=CUDA.zeros(Float32, D),
               chain_type=MCMCChains.Chains)
```

---

## Picking an AD backend on GPU

| Backend | GPU support | Restrictions | Recommended when |
|---|---|---|---|
| `AutoEnzyme()` | Via `pmcmc_matmul` / `pmcmc_dot` / `pmcmc_dotsum` and single-op-broadcast gradients | The restrictions above | You want Enzyme's reverse-mode performance and are willing to follow the rules above |
| `AutoMooncake(; config=nothing)` | Native (no wrappers) | Slower compile | You want plain Julia operators with no rewriting |
| `AutoZygote()` | Native (no wrappers) | ChainRules-based; reverse-only | You want plain Julia operators and are already on the ChainRules ecosystem |
| `AutoForwardDiff()` | Works if `logp` is `CuArray{Dual}`-compatible (broadcasts only, no `*`/`dot`) | No matmul; impractical for most GPU models | Smoke tests of element-wise targets only |

If you supply `hvp` / `hvp_batch` analytically, none of the above matters — those run as plain kernels and the sampler does not invoke a backend.
