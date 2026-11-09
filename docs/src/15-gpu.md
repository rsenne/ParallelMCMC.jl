# GPU execution

Every sampler allocates its working arrays with the same array type as `initial_params`, so the same sampler runs on CPU vectors and on supported device vectors.
To sample on an NVIDIA GPU, load CUDA with `using CUDA`, which activates ParallelMCMC's CUDA extension, and pass a `CuVector` as `initial_params`.

Changing the type of the initial state is not enough, though: the target density, its derivatives, and the selected AD backend must all be written in a device-compatible manner.

## Decide whether a GPU is worthwhile

A GPU helps only when each DEER iteration is expensive enough to amortize kernel launches and data movement.
Most of that work is evaluating the density, gradient, and Hessian-vector products at all `T` states of the trajectory.

| Regime | Reach for |
|---|---|
| Small `D` and a cheap target | CPU; launch overhead is likely to dominate. |
| Large `D` or a target built from substantial batched linear algebra | GPU; the model is more likely to keep the device occupied. |
| Many small independent chains | CPU threads; GPU chains do not currently share a device pool. |

Benchmark the model before and after moving it to the GPU.
Run each version once to compile it, then compare later calls with the same `T`, step size, and tolerances.

## Worked example: Bayesian logistic regression

This example fits a Bayesian logistic regression with a standard normal prior on the coefficients $\beta \in \mathbb{R}^D$, given $N$ observations.
Its cost grows with both $N$ and $D$, which is the regime where a GPU can help.
The log-posterior is

```math
\log p(\beta \mid y, X) = -\tfrac{1}{2}\|\beta\|^2 + \sum_{i=1}^{N} \bigl[ y_i \, x_i^\top \beta - \log(1 + e^{x_i^\top \beta}) \bigr],
```

and its gradient is $\nabla_\beta \log p = -\beta + X^\top \bigl(y - \sigma(X\beta)\bigr)$, where $\sigma$ is the logistic function.

The model computes $\log(1 + e^{z})$ as

```math
\log(1 + e^{z}) = \max(z, 0) + \log(1 + e^{-|z|}),
```

so that neither tail overflows in `Float32`.

### Synthetic data

```julia
using ParallelMCMC, Random, CUDA

const D = 50
const N = 1000
rng    = MersenneTwister(0)
β_true = randn(rng, Float32, D)
X_cpu  = randn(rng, Float32, N, D)
probs  = 1f0 ./ (1f0 .+ exp.(-(X_cpu * β_true)))
y_cpu  = Float32.(rand(rng, N) .< probs)

const X_gpu = CUDA.CuMatrix(X_cpu)
const y_gpu = CUDA.CuVector(y_cpu)
```

The model functions below capture `X_gpu` and `y_gpu`.
Declaring them `const` keeps those functions type-stable.

### Mooncake backend

With Mooncake, the model uses ordinary Julia operators:

```julia
using LinearAlgebra
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
                              T=16,
                              backend=ADTypes.AutoMooncake(; config=nothing))

chain = sample(model, sampler, 1_600; initial_params=CUDA.zeros(Float32, D))
```

The model has no hand-written HVP, so the sampler builds each HVP by differentiating `gradlogp` with Mooncake.
To check the fit, compare the posterior mean with the coefficients that generated the data:

```julia
using Statistics
β_post = mean(chain[:x])
norm(β_post - β_true) / norm(β_true)
```

### Enzyme restrictions

With `AutoEnzyme()` on the GPU, differentiating `*`, `dot`, or `sum` over `CuArray`s can abort Enzyme compilation with an error like:

```text
unsupported tag gc-transition for
  call i32 @cuMemcpyDtoHAsync_v2(...) [ "jl_roots"(...), "gc-transition"() ]
UNREACHABLE executed at .../Enzyme/GradientUtils.cpp:309      (signal 6)
```

ParallelMCMC exports three wrappers with their own Enzyme rules that avoid this failure:

```julia
pmcmc_matmul(A, B)  = A * B
pmcmc_dot(a, b)     = dot(a, b)
pmcmc_dotsum(A, B)  = sum(A .* B)
```

Use them in GPU code that Enzyme differentiates.
CPU code, and GPU code differentiated by Mooncake or Zygote, should keep the ordinary operators, which those backends already handle on `CuArray`s.
The wrappers only change how Enzyme differentiates these operations; they do not make CPU-only code run on the GPU.

Some `CuArray` gradients fail with the same error even with the wrappers.
Computing each broadcast in its own statement usually fixes this:

```julia
# Fails to compile with Enzyme on the GPU
gradlogp(β) = -β .- pmcmc_matmul(transpose(X), pmcmc_matmul(X, β)) ./ Float32(N)

# Compiles: one operation per line
function gradlogp(β)
    Xβ = pmcmc_matmul(X, β)
    Y  = pmcmc_matmul(transpose(X), Xβ)
    Y  = Y ./ Float32(N)
    Y  = Y .+ β
    return -Y
end
```

Mooncake does not have either restriction.

### Enzyme backend

The Enzyme version of the same model replaces `*` with `pmcmc_matmul` and `dot` with `pmcmc_dot`, and computes each broadcast in its own statement:

```julia
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
                              T=16,
                              backend=ADTypes.AutoEnzyme())

chain = sample(model, sampler, 1_600; initial_params=CUDA.zeros(Float32, D))
```

Both examples target the same posterior.
The Enzyme version is longer only because it follows the two restrictions above.

### The same model with a hand-written HVP

The Hessian of the log-posterior is

```math
\nabla^2 \log p = -I - X^\top \, \mathrm{diag}\bigl(\sigma(X\beta) \odot (1 - \sigma(X\beta))\bigr) \, X,
```

so

```math
H v = -v - X^\top \bigl[\sigma(X\beta) \odot (1 - \sigma(X\beta)) \odot (X v)\bigr].
```

Pass `hvp` and `hvp_batch` to the model and leave out the sampler's `backend`:

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

sampler = ParallelMALASampler(0.005f0; T=16)
```

The posterior is the same as before.
Only the source of the HVPs has changed.
The sampler now calls no AD backend, so this version can use the plain operators of the Mooncake version even if you would otherwise use Enzyme.

## Where the HVP comes from

The examples above pass a gradient function and a sampler `backend`, so the sampler builds each HVP by differentiating `gradlogp` with that backend.
`AutoEnzyme()` and `AutoForwardDiff()` push a tangent through `gradlogp`.
`AutoMooncake()` and `AutoZygote()` differentiate `x -> dot(gradlogp(x), v)` in reverse mode instead.
[Defining models](12-models.md) lists the other combinations, and when a hand-written HVP is worth writing.

!!! note "AD-derived batched gradients"
    If `grad_logdensity` is an AD backend and you leave out `grad_logdensity_batch`, the sampler differentiates `logdensity_batch` with that backend.
    `logdensity_batch` then has to satisfy the same backend restrictions as the rest of your model.
    Pass `grad_logdensity_batch` to avoid this.

## Reactant

`ADTypes.AutoReactant()` compiles the derivatives with [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl).
Load Reactant before calling `sample`:

```julia
using Reactant, ADTypes

model = DensityModel(logp, AutoReactant(), D)
sampler = ParallelMALASampler(0.005f0; T=16, backend=AutoReactant())
```

The log-density must be traceable by Reactant.
Use `AutoReactant()` for both the gradient and the HVP, or for neither, with one exception: a hand-written gradient can be combined with an `AutoReactant()` HVP backend.
`AutoReactant()` cannot appear inside a `SecondOrder`, and only its default mode is supported.

!!! warning "Captured arrays become constants"
    Reactant compiles arrays captured by the log-density into the derivatives as constants.
    If you change such an array after `sample` has started, the log-density sees the change but the derivatives do not.

!!! warning "Reactant chooses its own device"
    Reactant runs on its default client, whatever the type of `initial_params`.
    With a CUDA client, `CuArray` inputs are copied through pinned host memory and the results are copied back to the device.
    With the CPU client, both copies go through the host, and the sampler warns when it prepares the model.
    Call `Reactant.set_default_backend("gpu")` to select a GPU client when one is available.

## Choosing an AD backend on the GPU

| Backend | Model code on the GPU | Notes |
|---|---|---|
| `AutoEnzyme()` | `pmcmc_*` wrappers and one broadcast per statement | See [Enzyme restrictions](@ref) above. |
| `AutoMooncake(; config=nothing)` | Plain Julia operators | Slower to compile. |
| `AutoZygote()` | Plain Julia operators | `gradlogp` must not mutate arrays. |
| `AutoForwardDiff()` | Plain Julia operators | Cost grows with `D`; useful for small models and for checking other backends. |

None of this applies when you pass `hvp` and `hvp_batch` yourself, because the sampler then calls no AD backend.
