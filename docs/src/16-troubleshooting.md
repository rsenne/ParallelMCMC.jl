# Troubleshooting

## `ParallelMALASampler needs a Hessian-vector product`

The model has a hand-written gradient but no `hvp`, and the sampler has no `backend`.
Pass `hvp` to [`DensityModel`](@ref), or pass an AD backend such as `backend=AutoForwardDiff()` to [`ParallelMALASampler`](@ref).
For a model with batched functions, the matching error is "batched DEER requires `hvp_batch` or a sampler `backend`", and the same fixes apply.
See [Defining models](12-models.md) for how each choice computes the HVP.

## DEER did not converge in `maxiter` iterations

When a block reaches `maxiter` before it meets the tolerances, the sampler still returns its states as samples and logs one warning saying how many blocks this happened to.
Those samples are not exact MALA draws.

What the warning means depends on how `T` compares with `maxiter`.
Recall from [How the algorithm works](20-algorithms.md#Damping-and-clipping-the-Jacobian) that after $t$ iterations the first $t$ states of a block are exact, so in exact arithmetic the solve finishes within `T` iterations.
If `T` is larger than `maxiter`, the solve may just need more iterations than it was given.
In that case, raise `maxiter`, or see [The solve needs many iterations](@ref) below.

With the defaults, though, `T=64` is well below `maxiter=200`, so the solve should not run out of iterations.
The warning then points to a numerical problem instead, such as a log-density or gradient that returns `NaN` or `Inf` at some state the solve visits.
A quick check is to look for non-finite values in the returned log-densities:

```julia
any(!isfinite, chain[:logp])
```

Even when a block does not converge, its samples are often close to the sequential ones.
In the logistic regression experiments of [Zoltowski et al. (2025)](https://arxiv.org/abs/2508.18413), a chain that hit the iteration limit was still effectively identical to sequential MALA, and stopping after only 8 iterations often gave samples of similar quality.
How close they get depends on the problem, though, so it is worth comparing them with [`MALASampler`](@ref) before relying on them.

## The solve needs many iterations

The sampler does not report how many iterations each block takes, so this usually shows up as sampling that is slower than you expect.
There are three common causes.

The first is the step size, because low acceptance makes the solve much slower.
In our tests on a logistic regression with 25 parameters, a block of 4096 transitions took 23 to 31 iterations at about 80% acceptance, 37 to 42 at about 58%, and 57 to 71 at about 47%.
At 12% acceptance, the same block took between 465 and 1100.
The default target of [`AdaptiveMALASampler`](@ref), 57%, is a reasonable choice, so tuning the step size as in [Getting started](10-getting-started.md) is usually enough.
A higher acceptance rate speeds up the solve further, but the chain then moves less per sample.

The second is strongly correlated parameters.
DEER approximates each transition's Jacobian by its diagonal.
Ignoring the accept/reject step, the Jacobian of a MALA step is $I + \varepsilon \nabla^2 \log p(x)$, so the approximation is good when the Hessian is close to diagonal and poor when the parameters are strongly correlated.
Zoltowski et al. (2025) handle this by rotating the coordinates so that the Hessian is closer to diagonal.
Since the MALA proposal adds the same amount of noise in every direction, sampling $z$ from $p(Q z)$ for an orthogonal matrix $Q$ is the same sampler in rotated coordinates.
This means we can rotate the model ourselves:

```julia
using LinearAlgebra
Q = eigen(Symmetric(H)).vectors  # H is the Hessian of log p at a typical point

model_z = DensityModel(z -> logdensity(Q * z), z -> Q' * gradient(Q * z), dim;
                       hvp=(z, v) -> Q' * hvp(Q * z, Q * v))
```

`H` can come from `ForwardDiff.hessian` at the last state of a short sequential run.
We then sample `model_z` starting from `Q' * x0`, and map each sample back with $x = Q z$.
For a generalized linear model, the eigenvectors of $X^\top X$ also work, and the paper used those for its 768-dimensional sentiment model.
On our logistic regression, whose features were nearly uncorrelated, the rotation saved 20 to 30 percent of the iterations, and we expect it to help more when the parameters are strongly correlated.
The rotation only gives the same sampler without a mass matrix (`cholM`), and batched functions need the same rotation.

The third is a multimodal target.
There the local Jacobians can change sharply from one state to the next, which can make the solve unstable.
Damping the Jacobian with `jacobian_damping=0.5` helps here.
On a mixture of four Gaussians, Zoltowski et al. (2025) needed 50 iterations with damping, against 72 with clipping alone.
On targets where the solve is already stable, damping only slows it down.

## Parallel MALA is slower than sequential MALA

Parallel MALA does more total work than sequential MALA, so it is only faster when the hardware can do that extra work in parallel.
For a small, cheap target on a CPU, sequential MALA is usually faster.

On a GPU, the most common reason is a block that is too short.
When the GPU has room to spare, the solve replaces the `T` sequential steps of a block with one step per iteration.
With the default `T=64`, that saved only a factor of about 5 in our tests, so see [Choose `T` and the DEER controls](@ref) for how to pick a longer block.

Running many chains at once has the same effect, since the chains compete for the same hardware.
Zoltowski et al. (2025) found sequential MALA slightly faster than parallel MALA with 32 chains on one GPU (see [Run several chains](@ref)).

When comparing the two, time the complete `sample` call, and run each version once first so that compilation is not counted.

## Running out of memory

Each block keeps several $D \times T$ arrays, and batched model functions add their own, such as the $N \times T$ matrix $X B$ in a regression.
If the GPU runs out of memory, reduce `T` or the number of chains.
Zoltowski et al. (2025) ran into the same limit with 32 chains of 64,000 samples each.

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
Low acceptance also slows the DEER solve, as described in [The solve needs many iterations](@ref).

## GPU code reports scalar indexing or AD compilation errors

Indexing a `CuArray` element by element, as in `x[1]`, is an error on the GPU.
Write the model with whole-array operations and broadcasting instead.

AD compilation errors on the GPU are usually backend-specific.
See [GPU execution](15-gpu.md) for the restrictions of each backend.
