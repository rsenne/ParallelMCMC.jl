# Troubleshooting

## `ParallelMALASampler needs a Hessian-vector product`

The model has a hand-written gradient but no `hvp`, and the sampler has no `backend`.
Pass `hvp` to [`DensityModel`](@ref), or pass an AD backend such as `backend=AutoForwardDiff()` to [`ParallelMALASampler`](@ref).
For a model with batched functions, the matching error is "batched DEER requires `hvp_batch` or a sampler `backend`", and the same fixes apply.
See [Defining models](12-models.md) for how each choice computes the HVP.

## The DEER solve does not converge

The sampler does not report how many iterations each block needed.
A block that reaches `maxiter` is returned as samples without a warning.
Suspect the solve if sampling is much slower than you expect, or if the samples disagree with [`MALASampler`](@ref) on the same model.

First reduce `epsilon` or `T`.
If the model otherwise behaves well, lower `damping` from `0.5` toward `0.3`.
Do not loosen the tolerances until you have ruled out a derivative error.

## Parallel MALA is slower than sequential MALA

That is expected for small, cheap targets on a CPU.
Each DEER iteration evaluates the model and its derivatives at all `T` states, so parallel MALA does more total work than sequential MALA.
It is faster only when parallel hardware can absorb that extra work.

When comparing the two, benchmark the complete sampling call, and measure compilation separately from steady-state execution.

## The chain does not move or has many rejections

Run [`MALASampler`](@ref) or [`AdaptiveMALASampler`](@ref) on the same model first.
If they fail too, the problem is in the model rather than in DEER.
Compare a hand-written gradient with an AD gradient at a few random points:

```julia
using ForwardDiff
x = randn(dim)
ForwardDiff.gradient(logdensity, x) ≈ gradient(x)
```

If the gradient is correct, try a smaller step size.

## GPU code reports scalar indexing or AD compilation errors

Indexing a `CuArray` element by element, as in `x[1]`, is an error on the GPU.
Write the model with whole-array operations and broadcasting instead.

AD compilation errors on the GPU are usually backend-specific.
See [GPU execution](15-gpu.md) for the restrictions of each backend.
