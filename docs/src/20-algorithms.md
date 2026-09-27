# How the algorithm works

This page explains the implementation behind [`ParallelMALASampler`](@ref).
The main idea comes from Zoltowski et al. (2025) [^1]: turn a sequential Markov
recursion into a nonlinear system, then solve that system with parallel
operations.

You do not need this material to use the sampler. It is useful when choosing
`T`, diagnosing convergence, or working with the low-level DEER API.

## From a chain to a nonlinear system

Once its random draws have been fixed, a Markov chain is a deterministic
recursion:

```math
s_t = f_t(s_{t-1}), \qquad t = 1, \ldots, T.
```

The subscript on $f_t$ accounts for the random values used by transition $t$.
ParallelMCMC draws those values in advance and stores them in a *tape*.

Instead of applying the transitions one after another, collect the unknown
states into $S = (s_1, \ldots, s_T)$ and define

```math
r_t(S) = s_t - f_t(s_{t-1}).
```

The taped sequential trajectory is the root of $r(S)=0$. DEER searches for the
same root while exposing work across all $T$ transitions.

## One DEER iteration

Let $S^{(i)}$ be the current guess. Linearize each transition around the
previous state in that guess:

```math
J_t = \nabla f_t\!\left(s^{(i)}_{t-1}\right),
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

All $J_t$ and $u_t$ values depend only on the old guess, so they can be
evaluated together. The remaining affine recursion can be solved by an
associative scan rather than a serial loop.

With damping, the package replaces the raw update with

```math
S^{(i+1)} \leftarrow
(1 - \lambda) S^{(i)} + \lambda S^{(i+1)},
\qquad 0 < \lambda \leq 1.
```

This is the `damping` keyword. Smaller values often make difficult solves more
stable, at the cost of more iterations.

## Why the diagonal approximation matters

A dense $D \times D$ Jacobian at every transition is expensive to form, store,
and combine. The public sampler therefore uses a diagonal approximation:

```math
s^{(i+1)}_{d,t}
= a_{d,t} s^{(i+1)}_{d,t-1} + b_{d,t}.
```

Each parameter dimension is now a scalar affine recursion. This is often
called quasi-DEER.

`ParallelMALASampler` offers two ways to get the diagonal:

- `jacobian=:diag` computes it exactly with $D$ Jacobian-vector products. This
  is useful for small models and reference checks.
- `jacobian=:stoch_diag` uses a Hutchinson estimator. This is the default and
  usually the only practical choice in high dimensions.

For independent Rademacher vectors $z^{(k)}$, whose entries are equally likely
to be $-1$ or $1$,

```math
\operatorname{diag}(J_t)
\approx \frac{1}{K}\sum_{k=1}^{K}
z^{(k)} \odot J_t z^{(k)}.
```

One probe needs one Jacobian-vector product instead of one per dimension. The
`probes` keyword sets $K$. More probes reduce the estimator's noise but add
derivative work to every DEER iteration.

The diagonal is an approximation to the Newton direction, not a change to the
target transition. If the nonlinear solve converges, its fixed point still
satisfies the original taped recursion up to the requested numerical tolerance.

## The parallel affine scan

Represent one scalar affine map as a pair $(a, b)$ acting on $x$ by
$a x + b$. Composing two maps gives

```math
(a_2, b_2) \circ (a_1, b_1)
= (a_2 a_1,\; a_2 b_1 + b_2).
```

Composition is associative. A parallel-prefix scan can therefore combine the
maps for all prefixes in $O(\log T)$ dependent levels. Each level still does
work across the trajectory, so $O(\log T)$ describes the critical path of the
scan, not zero total work.

The implementation in
`ParallelMCMC.DEERScan.solve_affine_scan_diag!` uses array operations over
$D \times T$ matrices. The same code works with ordinary matrices and supported
device matrices.

## Applying the method to MALA

Using the convention in this package, a MALA proposal is

```math
\widetilde{x}_t = x_{t-1}
+ \varepsilon \nabla \log p(x_{t-1})
+ \sqrt{2\varepsilon}\,\xi_t,
```

followed by the usual Metropolis-Hastings accept/reject decision. The tape holds
the Gaussian noise $\xi_t$ and the uniform draw used by that decision.

The hard accept/reject indicator is not differentiable. DEER still needs a
useful local derivative, so the implementation uses a smooth sigmoid surrogate
for differentiation while keeping the actual decision fixed in the forward
evaluation. In other words, the sampler does not replace Metropolis-Hastings
with a soft acceptance rule; the relaxation only supplies a Jacobian for the
nonlinear solver.

The relevant implementation is
[`MALA.mala_step_surrogate_sigmoid`](@ref MALA.mala_step_surrogate_sigmoid).

## Convergence and sample delivery

After each update, DEER compares the largest elementwise change with a mixed
absolute and relative tolerance:

```math
\max |S^{(i+1)} - S^{(i)}|
\leq \texttt{tol_abs}
+ \texttt{tol_rel}\,\max |S^{(i+1)}|.
```

The solve stops when this condition is met or after `maxiter` iterations. Once
a block has been solved, its `T` columns are delivered as MCMC samples. If the
caller asks for more than `T` samples, the sampler starts another block from the
last state and draws a new tape. A final partial block is trimmed so that the
returned chain has the requested length.

For low-level callers, `DEER.solve(...; return_info=true)` returns convergence
information with the trajectory. When reusing a `DEERWorkspace`, the result is
copied by default so a later solve cannot overwrite it. Set `copy_result=false`
only when accepting a workspace-owned result is intentional.

## What controls performance?

The scan has a logarithmic dependency depth, but the whole sampler also pays
for model evaluations, derivative products, nonlinear iterations, memory
traffic, and compilation. The most important practical factors are:

- how efficiently the model evaluates a `T`-column batch;
- the number of DEER iterations needed for convergence;
- the Jacobian mode and number of probes;
- the trajectory length and parameter dimension;
- whether the hardware has enough parallel work to offset launch and setup
  costs.

This is why parallel MALA is not automatically faster than sequential MALA.
Measure both on the target and hardware that matter for your application.

## One block at a glance

1. Draw a tape of `T` MALA noise and uniform values.
2. Evaluate each taped transition at the current trajectory guess.
3. Estimate or compute each transition's Jacobian diagonal.
4. Solve the resulting affine recursion with a parallel scan.
5. Apply damping and test convergence; repeat if needed.
6. Return the solved states as samples and continue from the last one.

[^1]: Zoltowski, D. M., Wu, S., Gonzalez, X., Kozachkov, L., & Linderman,
    S. W. (2025). *Parallelizing MCMC Across the Sequence Length*. NeurIPS
    2025. [arXiv:2508.18413](https://arxiv.org/abs/2508.18413)
