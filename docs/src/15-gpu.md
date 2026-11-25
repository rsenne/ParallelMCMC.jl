# GPU execution

There are two ways to run ParallelMCMC on an NVIDIA GPU (AMD GPUs are not explicitly supported as of now).

With [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), the sampler keeps its arrays on the GPU as `CuArray`s and calls your model's functions on them.
This works on Linux and Windows.
The derivatives can be written by hand or come from an AD backend that supports `CuArray`s.

With [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), ParallelMCMC compiles each DEER iteration, including your log-density and its derivatives, into one XLA program that Reactant runs on the GPU.

Reactant only supports GPUs on Linux, so on Windows you have to run Julia inside WSL.

Which one to pick mostly depends on where your gradient comes from.
If you can write it by hand, CUDA.jl is the simpler route, and an AD backend can still compute the HVPs for you.
If you want the gradient from AD as well, use Reactant.
On `CuArray`s, only Zygote managed both the gradient and the HVP in our tests, and its first call was slow (see [HVPs from an AD backend](@ref)).

## Is a GPU worth it?

While Parallel MALA needs fewer sequential steps, it does more total work than sequential MALA and keeps all `T` states in memory at once.
To really reap the benefits, the GPU needs enough cores to run that work in parallel and enough memory to hold the whole block.
Most of that work is evaluating the log-density, its gradient, and Hessian-vector products (HVPs) at all `T` states of the trajectory.
Consequently, with small or simple models, you might find it hard to see real speedups with this method.
Large data, many parameters, and a long block `T` all help.
For a small, cheap target the CPU _can_ be faster, because kernel launches and copies between the host and the GPU dominate.
But the space of models where this is true likely also overlaps with the space of models that can be quickly evaluated using a sequential algorithm.

To compare the CPU and the GPU, run each version once to compile it, then time later calls with the same `T`, step size, and tolerances.

## The example model

Both sections below sample a Bayesian logistic regression with $N$ observations and a standard normal prior on the coefficients $\beta \in \mathbb{R}^D$:

```math
\log p(\beta \mid y, X) = -\tfrac{1}{2}\|\beta\|^2 + \sum_{i=1}^{N} \bigl[ y_i \, x_i^\top \beta - \log(1 + e^{x_i^\top \beta}) \bigr].
```

Its gradient and HVP are

```math
\nabla \log p = -\beta + X^\top \bigl(y - \sigma(X\beta)\bigr),
\qquad
\nabla^2 \log p \, v = -v - X^\top \bigl[\sigma(X\beta) \odot (1 - \sigma(X\beta)) \odot X v\bigr],
```

where $\sigma$ is the logistic function.
The code computes $\log(1 + e^{z})$ as $\max(z, 0) + \log(1 + e^{-|z|})$ so that neither tail overflows in `Float32`.

```julia
using ParallelMCMC, Random

const D = 50
const N = 1000
rng = MersenneTwister(0)
β_true = randn(rng, Float32, D)
X_cpu = randn(rng, Float32, N, D)
y_cpu = Float32.(rand(rng, N) .< 1 ./ (1 .+ exp.(-(X_cpu * β_true))))

softplus(z) = max(z, zero(z)) + log1p(exp(-abs(z)))
sigmoid(z) = 1 / (1 + exp(-z))
```

## With CUDA.jl

For the CUDA.jl route, load CUDA and pass a `CuVector` as `initial_params`.
Every sampler then allocates its working arrays on the GPU, and your functions receive `CuArray`s.
It is important to write them with whole-array operations and broadcasting, because indexing a `CuArray` one element at a time will error.

### Hand-written derivatives

Hand-written derivatives are the most reliable choice on the GPU, since no AD backend is involved, and will generally also be faster.
Just be sure your hand-written functions follow Julia and GPU best practices.

```julia
using CUDA

const X = CuMatrix(X_cpu)
const y = CuVector(y_cpu)

logp(β) = (z = X * β; sum(y .* z .- softplus.(z)) - sum(abs2, β) / 2)
gradlogp(β) = X' * (y .- sigmoid.(X * β)) .- β
hvp(β, v) = (s = sigmoid.(X * β); -(X' * (s .* (1 .- s) .* (X * v))) .- v)

function logp_batch(B)
    Z = X * B
    return vec(sum(y .* Z .- softplus.(Z); dims=1)) .- vec(sum(abs2, B; dims=1)) ./ 2
end
gradlogp_batch(B) = X' * (y .- sigmoid.(X * B)) .- B
hvp_batch(B, V) = (S = sigmoid.(X * B); -(X' * (S .* (1 .- S) .* (X * V))) .- V)

model = DensityModel(logp, gradlogp, D;
                     hvp=hvp,
                     logdensity_batch=logp_batch,
                     grad_logdensity_batch=gradlogp_batch,
                     hvp_batch=hvp_batch)
sampler = ParallelMALASampler(0.005f0; T=16)
chain = sample(model, sampler, 1_600; initial_params=CUDA.zeros(Float32, D))
```

There are a few things worth pointing out here.
We declare `X` and `y` as `const` so that the functions that capture them stay type-stable.
We also write batched versions of every function, which matters most on the GPU: with them, each DEER iteration makes a few large calls on $D \times T$ matrices instead of $T$ small ones.
All three batched functions start by computing $X B$, so if you want to compute it only once, see [Value, gradient, and HVP in one call](@ref).

To check the fit, compare the posterior mean with the coefficients that generated the data:

```julia
using LinearAlgebra, Statistics
β_post = mean(chain[:x])
norm(β_post - β_true) / norm(β_true)
```

### HVPs from an AD backend

If you would rather not write the HVP, keep the hand-written gradients and pass an AD backend to the sampler.
The sampler then differentiates `gradlogp` and `gradlogp_batch` to get each HVP:

```julia
using ADTypes, Zygote

model = DensityModel(logp, gradlogp, D;
                     logdensity_batch=logp_batch,
                     grad_logdensity_batch=gradlogp_batch)
sampler = ParallelMALASampler(0.005f0; T=16, backend=AutoZygote())
```

We ran this model with each backend on an RTX 5090, with CUDA.jl 6.5 on Julia 1.13 and Enzyme also on Julia 1.11:

| `backend` | Works on `CuArray`s | Notes |
|---|---|---|
| `AutoForwardDiff()` | Yes | Pushes one tangent through the gradient, so it costs about as much as a hand-written HVP. |
| `AutoZygote()` | Yes | The gradient must not mutate arrays. |
| `AutoMooncake()` | Yes | The first `sample` call took more than twice as long as with the other two. |
| `AutoEnzyme()` | Only on Julia 1.11, with changes | See [Enzyme](@ref) below. |

With all three working backends, the samples matched those from the hand-written HVPs to `Float32` rounding.

Getting the gradient itself from AD (without Reactant) on the GPU is harder.
Zygote was the only backend that managed both the gradient and the HVP, and its first `sample` call took about 100 s.
Mooncake's gradient works fine for [`MALASampler`](@ref) and [`AdaptiveMALASampler`](@ref), but its HVP hit a scalar indexing error, and ForwardDiff and Enzyme failed on the gradient itself.
As AD support for `CuArray`s improves, this will hopefully become less of an issue.
Until then, use Reactant.

### Enzyme

On Julia 1.11, Enzyme can compute HVPs from a hand-written gradient, but only with two changes.
First, the gradient has to use `pmcmc_matmul` instead of `*`.
This wrapper, along with `pmcmc_dot` for `dot` and `pmcmc_dotsum` for `sum(A .* B)`, has its own Enzyme rule, so Enzyme does not try to differentiate the cuBLAS call inside it.
Second, runtime activity has to be on, or the model fails with an `EnzymeRuntimeActivityError`:

```julia
using ADTypes, Enzyme

gradlogp(β) = pmcmc_matmul(X', y .- sigmoid.(pmcmc_matmul(X, β))) .- β
gradlogp_batch(B) = pmcmc_matmul(X', y .- sigmoid.(pmcmc_matmul(X, B))) .- B

model = DensityModel(logp, gradlogp, D;
                     logdensity_batch=logp_batch,
                     grad_logdensity_batch=gradlogp_batch)
mode = Enzyme.set_runtime_activity(Enzyme.Forward)
sampler = ParallelMALASampler(0.005f0; T=16, backend=AutoEnzyme(; mode=mode))
```

With `*` instead of `pmcmc_matmul`, it fails even with runtime activity.
CPU code and the other backends should keep the ordinary operators.

## With Reactant

Using Reactant is as simple as loading it and passing `AutoReactant()` as the gradient (you can also pass a hand-written gradient and `AutoReactant()` as the HVP).
We also give the model a `logdensity_batch`, which lets the sampler compile each whole DEER iteration into one XLA program.
Without it, the sampler calls a compiled gradient and HVP once per state, and in our test that was about 15 times slower.

```julia
using ParallelMCMC, Reactant, ADTypes

logp(β) = (z = X_cpu * β; sum(y_cpu .* z .- softplus.(z)) - sum(abs2, β) / 2)
function logp_batch(B)
    Z = X_cpu * B
    return vec(sum(y_cpu .* Z .- softplus.(Z); dims=1)) .- vec(sum(abs2, B; dims=1)) ./ 2
end

model = DensityModel(logp, AutoReactant(), D; logdensity_batch=logp_batch)
sampler = ParallelMALASampler(0.005f0; T=16)
chain = sample(model, sampler, 1_600; initial_params=zeros(Float32, D))
```

It is important that the data and `initial_params` stay regular arrays.
A `CuArray` as `initial_params` fails, because the sampler also calls `logp` directly, and Reactant cannot compile functions that capture `CuArray`s.

Notice that we never wrote a gradient or an HVP.
ParallelMCMC traces your log-density and differentiates it with Enzyme inside Reactant, so both come from `logp` and `logp_batch`.
For the same reason the sampler needs no `backend`; you can pass `backend=AutoReactant()`, but it already uses Reactant for the HVP.
The catch is that the log-density must be traceable by Reactant, which rules out, for example, branches on the values of `β`.

### Compile time

Compilation happens more often than you might expect.
Every `sample` call compiles the DEER iteration again, although the chains of one `MCMCThreads` or `MCMCSerial` call share it.
The first call in a new Julia session also has to compile Reactant's own tracing code.
For a logistic regression the size of the German Credit data ($N = 1000$, $D = 25$) on an RTX 5090, the first `sample` call took about 50 s, and later calls spent about 2 s compiling.

### Limitations

Use `AutoReactant()` for both the gradient and the HVP, or for neither.
The one exception is a hand-written gradient with `backend=AutoReactant()`, which works but keeps the uncompiled DEER loop.
`AutoReactant()` cannot appear inside a `SecondOrder`, and only its default mode is supported.

Some settings also turn off the compiled iteration, even when the model has `logdensity_batch`: `jacobian=:diag`, or a hand-written `grad_logdensity_batch`, `hvp_batch`, or `value_grad_hvp_batch`.
Any of these sends the sampler back to the uncompiled DEER loop.

A compiled run does not exactly reproduce an uncompiled one with the same `rng`.
The Hutchinson probes come from Reactant's random number generator, seeded once per block from `rng`.
The DEER solution does not depend on the probes, so the first block matches the uncompiled loop up to the DEER tolerance.
Drawing the seed changes what `rng` produces afterwards, so later blocks differ.

!!! warning "Captured arrays become constants"
    Reactant compiles arrays that your functions capture, such as `X_cpu` above, into the program as constants.
    Each `sample` call compiles again, so it sees changes you make to them between calls, but not changes made while it runs.
