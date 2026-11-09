# How the algorithm works

This page explains the implementation behind [`ParallelMALASampler`](@ref).
The main idea comes from Zoltowski et al. (2025) [^1]: turn a sequential Markov recursion into a nonlinear system, then solve that system with parallel operations.

You do not need this material to use the sampler.
It is useful when choosing `T`, diagnosing convergence, or working with the low-level DEER API.

## Overview

For each block of `T` transitions, the sampler:

1. Draws a tape of `T` MALA noise and uniform values.
2. Evaluates each taped transition at the current trajectory guess.
3. Estimates or computes each transition's Jacobian diagonal.
4. Solves the resulting affine recursion with a parallel scan.
5. Tests convergence, repeating from step 2 if needed.
6. Returns the solved states as samples and continues from the last one.

The sections below explain each step.

## From a chain to a nonlinear system

Once its random draws have been fixed, a Markov chain is a deterministic recursion:

```math
s_t = f_t(s_{t-1}), \qquad t = 1, \ldots, T.
```

The subscript on $f_t$ accounts for the random values used by transition $t$.
ParallelMCMC draws those values in advance and stores them in a *tape*.

Instead of applying the transitions one after another, collect the unknown states into $S = (s_1, \ldots, s_T)$ and define

```math
r_t(S) = s_t - f_t(s_{t-1}).
```

The taped sequential trajectory is the root of $r(S)=0$.
DEER searches for the same root while exposing work across all $T$ transitions.

## One DEER iteration

Let $S^{(i)}$ be the current guess.
Linearize each transition around the previous state in that guess:

```math
J_t = \frac{\partial f_t}{\partial s}\!\left(s^{(i)}_{t-1}\right),
```

```math
u_t = f_t\!\left(s^{(i)}_{t-1}\right)
      - J_t s^{(i)}_{t-1}.
```

The next trajectory solves the affine recursion

```math
s^{(i+1)}_t = J_t s^{(i+1)}_{t-1} + u_t,
\qquad s^{(i+1)}_0 = s_0.
```

All $J_t$ and $u_t$ values depend only on the old guess, so they can be evaluated together.
The remaining affine recursion can be solved by an associative scan rather than a serial loop.

### Damping and clipping the Jacobian

When many diagonal entries of $J_t$ exceed 1 in magnitude, the affine recursion amplifies errors along the trajectory, and the iterates can wander far from the solution or overflow.
Zoltowski et al. (2025) [^1] handle this by changing the Jacobian estimate before the scan:

```math
\hat{J}_t = \operatorname{clip}\bigl(k \operatorname{diag}(J_t),\, -c,\, c\bigr),
\qquad u_t = f_t\!\left(s^{(i)}_{t-1}\right) - \hat{J}_t s^{(i)}_{t-1}.
```

Scaling by $k \in (0, 1]$ is the scale-ELK method of Gonzalez et al. (2024) [^2].
Because $u_t$ uses the same $\hat{J}_t$, the fixed point is still the sequential trajectory, and after $t$ iterations the first $t$ states are exact.
In exact arithmetic the solve therefore finishes in at most $T$ iterations for any $k$ and $c$.

These are the `jacobian_damping` ($k$) and `jacobian_clip` ($c$) keywords.
[`ParallelMALASampler`](@ref) defaults to $k = 1$ and $c = 1$, like the reference implementation, while `DEER.solve` defaults to $c = \infty$.
Clipping changes nothing when every entry is already inside $[-c, c]$, but $k < 1$ slows solves that were already stable.

The older `damping` keyword relaxes the iterate instead, as $S^{(i+1)} \leftarrow (1 - \lambda) S^{(i)} + \lambda S^{(i+1)}$.
That loses the $T$-iteration bound, so `damping` is deprecated.

## Why the diagonal approximation matters

A dense $D \times D$ Jacobian at every transition is expensive to form, store, and combine.
The public sampler therefore uses a diagonal approximation:

```math
s^{(i+1)}_{d,t}
= a_{d,t} s^{(i+1)}_{d,t-1} + b_{d,t}.
```

Each parameter dimension is now a scalar affine recursion.
This is often called quasi-DEER.

`ParallelMALASampler` offers two ways to get the diagonal:

- `jacobian=:diag` computes it exactly with $D$ Jacobian-vector products, which is useful for small models and reference checks.
- `jacobian=:stoch_diag`, the default, uses a Hutchinson estimator and is usually the only practical choice in high dimensions.

For independent Rademacher vectors $z^{(k)}$, whose entries are equally likely to be $-1$ or $1$,

```math
\operatorname{diag}(J_t)
\approx \frac{1}{K}\sum_{k=1}^{K}
z^{(k)} \odot J_t z^{(k)}.
```

One probe needs one Jacobian-vector product instead of one per dimension.
The `probes` keyword sets $K$.
More probes reduce the estimator's noise but add derivative work to every DEER iteration.

The diagonal approximates the Newton direction; it does not change the target transition.
If the nonlinear solve converges, its fixed point still satisfies the original taped recursion up to the requested tolerance.

## The parallel affine scan

Represent one scalar affine map as a pair $(a, b)$ acting on $x$ by $a x + b$.
Composing two maps gives

```math
(a_2, b_2) \circ (a_1, b_1)
= (a_2 a_1,\; a_2 b_1 + b_2).
```

Composition is associative, so a parallel-prefix scan can combine the maps for all prefixes in $O(\log T)$ dependent levels.
Each level still does work across the whole trajectory: $O(\log T)$ is the length of the critical path, not the total work.

`DEER.DEERScan.solve_affine_scan_diag!` implements the scan with array operations on $D \times T$ matrices, so the same code runs on CPU and GPU arrays.

## Applying the method to MALA

Using the convention in this package, a MALA proposal is

```math
\widetilde{x}_t = x_{t-1}
+ \varepsilon \nabla \log p(x_{t-1})
+ \sqrt{2\varepsilon}\,\xi_t,
```

followed by the usual Metropolis-Hastings accept/reject decision.
The tape holds the Gaussian noise $\xi_t$ and the uniform draw used by that decision.

The hard accept/reject indicator is not differentiable, but DEER needs a local derivative of each transition.
The implementation therefore differentiates a smooth sigmoid surrogate of the decision, while the forward evaluation keeps the exact accept/reject.
The samples still come from Metropolis-Hastings; the surrogate only supplies a Jacobian for the nonlinear solver.

The relevant implementation is [`MALA.mala_step_surrogate_sigmoid`](@ref MALA.mala_step_surrogate_sigmoid).

## Convergence and sample delivery

After each update, DEER compares the largest elementwise change with a mixed absolute and relative tolerance:

```math
\max |S^{(i+1)} - S^{(i)}|
\leq \texttt{tol_abs}
+ \texttt{tol_rel}\,\max |S^{(i+1)}|.
```

The solve stops when this condition holds or after `maxiter` iterations.
In both cases the block's `T` columns become samples.
A block that stops at `maxiter` is not an exact MALA trajectory, so `sample` logs a warning with the number of such blocks, and `DEER.solve` warns unless `warn=false`.
If you ask for more than `T` samples, the sampler starts the next block from the last state with a new tape.
The last block is trimmed so that the chain has the requested length.

`DEER.solve(...; return_info=true)` also returns `info.converged` and `info.iters`, which you can use to check a solve directly.

## What controls performance?

The scan needs only $O(\log T)$ dependent steps, but every DEER iteration also evaluates the model and its derivatives at all $T$ states.
The total cost therefore depends on how efficiently the model handles a `T`-column batch, how many iterations the solve needs, the Jacobian mode and number of probes, and the sizes $T$ and $D$.
Parallel hardware hides this cost only when there is enough of it to keep the device busy.

[^1]: Zoltowski, D. M., Wu, S., Gonzalez, X., Kozachkov, L., & Linderman,
    S. W. (2025). *Parallelizing MCMC Across the Sequence Length*. NeurIPS
    2025. [arXiv:2508.18413](https://arxiv.org/abs/2508.18413)

[^2]: Gonzalez, X., Warrington, A., Smith, J. T. H., & Linderman, S. W.
    (2024). *Towards Scalable and Stable Parallelization of Nonlinear RNNs*.
    NeurIPS 2024. [arXiv:2407.19115](https://arxiv.org/abs/2407.19115)
