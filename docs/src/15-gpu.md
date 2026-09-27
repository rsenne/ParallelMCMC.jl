# GPU execution

[`ParallelMALASampler`](@ref) follows the storage of `initial_params`, so the
same sampler can work with CPU vectors or supported device vectors. Moving the
initial state is not enough, though: the target density, its derivatives, and
the selected AD backend must all support the device.

CUDA is an optional dependency. `using CUDA` activates ParallelMCMC's extension
for `CuArray` storage without making CUDA part of a CPU-only installation.

---

## Decide whether a GPU is worthwhile

A GPU helps only when each DEER iteration provides enough work to amortize
kernel launches and data movement. The important workload is the evaluation of
batched densities, gradients, and Hessian-vector products across a `T`-column
trajectory.

| Regime | Reach for |
|---|---|
| Small `D` and a cheap target | CPU; launch overhead is likely to dominate. |
| Large `D` or a target built from substantial batched linear algebra | GPU; the model is more likely to keep the device occupied. |
| Many small independent chains | CPU threads; GPU chains do not currently share a device pool. |

A useful rule is to benchmark before and after moving the model. Include one
run for compilation, then compare warmed-up sampling calls at the same `T`,
step size, and tolerances.

---

## Current limitations

### Turing models are CPU-only

The Turing constructor and its DynamicPPL/LogDensityProblems path currently
execute with CPU vectors. They cannot run a `@model` directly against a
`CuArray`.

For GPU sampling, write the density and derivatives with GPU-compatible array
operations and pass them to [`DensityModel`](@ref), as in the example below.

### Enzyme needs ParallelMCMC's linear-algebra wrappers

`ParallelMCMC` exports three thin wrappers:

```julia
pmcmc_matmul(A, B)  = A * B
pmcmc_dot(a, b)     = dot(a, b)
pmcmc_dotsum(A, B)  = sum(A .* B)
```

ParallelMCMC defines Enzyme rules for these wrappers without changing the rules
for Julia's base operators. Use them in GPU code differentiated by Enzyme to
avoid failures such as:

```text
unsupported tag gc-transition for
  call i32 @cuMemcpyDtoHAsync_v2(...) [ "jl_roots"(...), "gc-transition"() ]
UNREACHABLE executed at .../Enzyme/GradientUtils.cpp:309      (signal 6)
```

Use the wrappers when:

- You are on GPU **and** using `AutoEnzyme()` as the AD backend.

Keep the ordinary operators when:

- CPU code: plain `*`, `dot`, `sum` are faster and clearer.
- GPU code with `AutoMooncake(; config=nothing)`: Mooncake's own CUDA extension handles these operations natively.
- GPU code with `AutoZygote()`: Zygote uses ChainRules adjoints, which have cuBLAS-backed rules for `*` / `dot` / `sum` on `CuArray`.

The wrappers are an Enzyme-specific compatibility layer; they do not make an
otherwise CPU-only function GPU compatible.

### Enzyme may need broadcasts split into separate expressions

Some `CuArray` gradients fail during Enzyme compilation with the same `gc-transition` error. If you encounter it, try splitting broadcasts into separate expressions, as in this example:

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
using ParallelMCMC, Random, CUDA

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
using ParallelMCMC, FlexiChains
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
               chain_type=VNChain)
```

Posterior mean recovery error `‖β_post − β_true‖ / ‖β_true‖` should land in the 0.1–0.2 range after a few hundred post-warmup samples.

### Enzyme backend (requires `pmcmc_matmul`)

Same model, with the GPU-Enzyme restrictions applied: every `*` becomes `pmcmc_matmul`, every `dot` becomes `pmcmc_dot`, and every gradient broadcast is expanded into single-op stages:

```julia
using ParallelMCMC, FlexiChains
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
               chain_type=VNChain)
```

Both examples describe the same posterior. The Enzyme version separates operations to avoid the compilation failures above.

---

## The AD-HVP fallback (and when to write your own)

DEER needs a Hessian–vector product $H v$ at every Newton step.  `DensityModel` accepts both `gradlogp` and (optionally) `hvp` / `hvp_batch`.  Depending on what you supply, the sampler does one of two things:

- **You supply `hvp` / `hvp_batch`.**  These run as plain kernels.  The AD backend is never invoked for HVPs.
- **You only supply `gradlogp` / `grad_logdensity_batch`.**  The sampler builds the HVP by differentiating your gradient — either a forward-mode pushforward of `gradlogp` ([`ForwardOnGrad`](https://github.com/rsenne/ParallelMCMC.jl/blob/main/src/DEER/DEER.jl), the default for most backends) or a reverse-mode gradient of `x -> dot(gradlogp(x), v)` ([`ReverseOnGrad`](https://github.com/rsenne/ParallelMCMC.jl/blob/main/src/DEER/DEER.jl), used for `AutoMooncake` and `AutoZygote`).  This is the **AD-HVP fallback**, and it is what the logistic-regression example above uses.

!!! note "A backend in `grad_logdensity` reaches `logdensity_batch` too"
    The batched path needs a batched gradient, and derives one from `logdensity_batch` when `grad_logdensity` is a backend.  That puts `logdensity_batch` under the same restrictions as the rest of your AD-visible code.  Supply `grad_logdensity_batch` to avoid it.

### Reactant HVPs

`ADTypes.AutoReactant()` provides a compiled HVP path through [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl). Load Reactant before preparing the model:

```julia
using Reactant, ADTypes

model = DensityModel(logp, AutoReactant(), D)
sampler = ParallelMALASampler(0.005f0; T=16, backend=AutoReactant())
```

!!! warning "Reactant constraints"
    **Captured arrays are frozen at compile time.** Mutating them after preparation does not change the compiled derivative. Pass mutable data as an argument. **Reactant also chooses the execution device independently of the input array.** With Reactant's CPU client, `CuArray` inputs round-trip through the host. The sampler warns about this when it prepares the model. Select a GPU client with `Reactant.set_default_backend` when available.

- The log-density must be Reactant-traceable. DynamicPPL-built log-densities are not.
- Across an HVP's two AD passes, use `AutoReactant()` for both or neither. A hand-written gradient may pair with it. `SecondOrder` cannot contain `AutoReactant()`.
- Only the default `AutoReactant()` mode is supported.

### When the fallback is the right call

- **Complex or composed models.**  Bayesian neural nets, hierarchical models with many transformations, mixtures, or anything where the Hessian has no convenient closed form.  Deriving and maintaining `hvp` by hand for these is error-prone; AD removes a whole class of bugs.
- **Prototyping.**  You want a correct sampler running before you optimize.  Drop in `gradlogp`, lean on AD, profile later.
- **Operations that AD libraries handle for free but are tedious to differentiate by hand.**  Special functions, link functions, log-sum-exp, normalizing constants of standard distributions, etc.
- **You want backend flexibility.**  With only `gradlogp` you can swap `AutoMooncake` ↔ `AutoEnzyme` ↔ `AutoZygote` to compare without rewriting model code.

### When to write your own analytical HVP

- **The HVP has a clean closed form.**  Quadratic priors, Gaussian likelihoods, GLMs (logistic, Poisson, probit) — the second derivative is a known function of intermediate quantities you already compute in `gradlogp`.  A few extra lines and you skip the AD pipeline entirely.
- **Performance matters and the AD compile is heavy.**  Enzyme and Mooncake both pay a one-shot compilation cost on the user's gradient.  For long-running chains this amortizes, but for many short runs the analytical HVP wins.
- **You're hitting AD-backend-specific GPU restrictions.** The [Enzyme limitations](#enzyme-needs-parallelmcmcs-linear-algebra-wrappers) above (`pmcmc_*` wrappers, staged broadcasts) only matter when the AD backend is invoked. Supplying an analytical HVP sidesteps them: your `gradlogp` and `hvp` can use plain `*`, `dot`, and `sum`, and the sampler's `backend` can be omitted because no AD is invoked.
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

sampler = ParallelMALASampler(0.005f0; T=16, damping=0.5f0)
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
