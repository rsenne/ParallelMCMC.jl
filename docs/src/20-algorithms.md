# Algorithm Details

This page describes the mathematics behind the DEER algorithm, the stochastic diagonal Jacobian estimator, and how both are applied to MALA.  The primary reference is Zoltowski et al. (2025) [^1].

## The core idea

A Markov chain can be written as a nonlinear recursion

```math
s_t = f_t(s_{t-1}), \quad t = 1, \ldots, T,
```

where $f_t$ is the transition kernel (e.g., one MALA step) and the subscript $t$ indicates that the kernel may depend on pre-generated randomness (noise vectors, uniform draws) that is fixed before solving.  Generating this randomness ahead of time is called **taping** the chain.

Treating the full trajectory $s_{1:T}$ as the unknown and defining the residual

```math
r_t(s_{1:T}) = s_t - f_t(s_{t-1}),
```

finding the trajectory is equivalent to solving $r(s_{1:T}) = 0$.  Sequentially running the chain is one way to solve this; DEER is another — and a parallelisable one.

---

## DEER: Newton's method for nonlinear recursions

Apply one step of Newton's method to the residual $r = 0$ starting from a trajectory guess $S^{(i)}$:

```math
s^{(i+1)}_t = J_t \, s^{(i+1)}_{t-1} + \underbrace{\bigl[f_t(s^{(i)}_{t-1}) - J_t \, s^{(i)}_{t-1}\bigr]}_{u_t},
```

where $J_t = \nabla_{s} f_t(s^{(i)}_{t-1})$ is the Jacobian of the transition at the current guess.

The key observation is that once $J_t$ and $u_t$ are computed from the previous iterate $S^{(i)}$ — which can be done **in parallel over $t$** — the update is a **linear** time-varying recursion

```math
s^{(i+1)}_t = J_t \, s^{(i+1)}_{t-1} + u_t, \quad s^{(i+1)}_0 = s_0,
```

which can itself be solved in $O(\log T)$ steps via a parallel prefix (associative) scan.

The full DEER loop repeats until the change between iterates falls below a tolerance:

```math
\max_t \| s^{(i+1)}_t - s^{(i)}_t \|_\infty \;\leq\; \delta_\text{abs} + \delta_\text{rel} \cdot \max_t \| s^{(i+1)}_t \|_\infty.
```

Newton's method converges **quadratically** once near the fixed point, so the number of iterations scales only logarithmically with $T$ in practice.

---

## The parallel prefix scan

When $J_t$ is diagonal (quasi-DEER), the linear recursion decouples dimension-by-dimension into $D$ independent scalar affine recursions

```math
s^{(i+1)}_{d,t} = a_{d,t} \, s^{(i+1)}_{d,t-1} + b_{d,t},
```

where $a_{d,t} = [J_t]_{dd}$ and $b_{d,t} = u_{d,t}$.  Written in matrix form: given $A, B \in \mathbb{R}^{D \times T}$, solve

```math
S[:,t] = A[:,t] \odot S[:,t-1] + B[:,t], \quad S[:,0] = s_0.
```

The associative operator for combining two adjacent segments $(\alpha_1, \beta_1)$ and $(\alpha_2, \beta_2)$ is

```math
(\alpha_2, \beta_2) \circ (\alpha_1, \beta_1) = (\alpha_2 \odot \alpha_1,\; \alpha_2 \odot \beta_1 + \beta_2).
```

This associativity means the recurrence can be solved by an inclusive parallel-prefix scan in $O(\log T)$ levels, each level consisting of a single broadcast over all $T$ columns and no per-timestep loops. The implementation in `ParallelMCMC.DEERScan.solve_affine_scan_diag!` is array-type-agnostic and runs identically on CPU `Matrix` and GPU `CuMatrix`. The CPU path is mainly useful for correctness checks; without a GPU or substantial parallel hardware, sequential MCMC is usually faster wall-clock.

---

## Jacobian variants

### Full DEER

Use the full $D \times D$ Jacobian $J_t$ at each timestep.  The linear recursion becomes a sequence of dense matrix multiplications, solved sequentially (no scan shortcut for the general case).  Cost per iteration: $O(TD^3)$.  Memory: $O(TD^2)$.  Accurate but impractical for large $D$.

This package currently exposes the diagonal scan variants below as `jacobian` modes; full-matrix DEER is discussed here for context but is not available through the `jacobian` keyword.

### Quasi-DEER (`:diag`)

Replace $J_t$ with $\mathrm{diag}(J_t)$, retaining only the diagonal.  The recursion reduces to the scalar affine scan described above, solved in $O(TD \log T)$ total work.  The exact diagonal is computed with `D` Jacobian-vector products.  This mode is useful for low-dimensional checks and reference runs.

### Stochastic quasi-DEER (`:stoch_diag`)

Computing the exact diagonal of $J_t$ requires a full Jacobian (or $D$ JVPs), which is expensive in high dimensions.  The **Hutchinson–Rademacher estimator** provides an unbiased approximation using only JVPs:

```math
\mathrm{diag}(J_t) \approx \frac{1}{K} \sum_{k=1}^K z^{(k)} \odot J_t z^{(k)},
\quad z^{(k)}_i \overset{\text{iid}}{\sim} \mathrm{Rademacher}(\pm 1).
```

Each probe $z^{(k)}$ costs a single Jacobian-vector product (one forward-mode or reverse-mode pass), so the total cost is $O(KTD)$ — linear in $D$ and $T$.  In the limit $K \to \infty$ the estimate converges to the true diagonal.  This is the default mode.  See Zoltowski et al. (2025) [^1] for convergence analysis.

In practice $K=1$ or $K=2$ probes works well; controlled by the `probes` argument of [`ParallelMALASampler`](@ref).

---

## Applying DEER to MALA

One step of MALA with pre-drawn noise $(\xi_t, u_t)$ is

```math
\tilde{x}_t = x_{t-1} + \varepsilon \nabla \log p(x_{t-1}) + \sqrt{2\varepsilon}\, \xi_t,
```

```math
x_t = g_t \, \tilde{x}_t + (1 - g_t)\, x_{t-1}, \quad g_t = \mathbf{1}[\log u_t < \log \alpha_t],
```

where $\log \alpha_t = \log p(\tilde{x}_t) + \log q(x_{t-1} \mid \tilde{x}_t) - \log p(x_{t-1}) - \log q(\tilde{x}_t \mid x_{t-1})$ is the Metropolis log-acceptance ratio.

### The stop-gradient trick

The indicator $g_t \in \{0, 1\}$ is not differentiable, so $\nabla_{x_{t-1}} x_t$ is undefined at accept/reject boundaries.  DEER needs this Jacobian to form the Newton update.

The solution is a **surrogate step** used only during Jacobian computation (the *forward* step always uses the exact indicator):

```math
x_t^\text{surrogate} = \hat{g}_t \, \tilde{x}_t + (1 - \hat{g}_t)\, x_{t-1},
\quad \hat{g}_t = \sigma(\log \alpha_t - \log u_t) + \underbrace{\mathrm{sg}\!\left(\mathbf{1}[\cdots] - \sigma(\cdots)\right)}_{\text{stop-gradient}},
```

where $\sigma$ is the logistic function.  The stop-gradient term makes $\hat{g}_t$ equal to the exact indicator in the forward pass while routing gradients through $\sigma$ during the backward pass.  This gives a well-defined, smooth Jacobian whose value at the operating point equals that of the relaxed step.

In the implementation, the accept indicator $g_t$ is pre-computed from the previous-iterate state and passed as a **frozen constant** to the surrogate ([`MALA.mala_step_surrogate_sigmoid`](@ref MALA.mala_step_surrogate_sigmoid)), so differentiation never touches the discontinuity.

### Summary of the DEER–MALA loop

1. **Tape generation.** Draw $T$ noise pairs $(\xi_t, u_t)$ and store them.
2. **Jacobian and offset computation (parallel over $t$).** For each $t$, evaluate $f_t$ at the current guess $s^{(i)}_{t-1}$ to get $u_t$, and differentiate the surrogate to get $\mathrm{diag}(J_t)$ (or the Hutchinson approximation thereof).
3. **Parallel scan.** Solve the diagonal linear recursion $S^{(i+1)}[:,t] = A[:,t] \odot S^{(i+1)}[:,t-1] + B[:,t]$ in $O(\log T)$ levels.
4. **Convergence check.** If the change is below tolerance, return $S^{(i+1)}$; otherwise go to step 2.
5. **Sample delivery.** Return the $T$ columns of the converged trajectory as individual MCMC samples.

For low-level callers using `DEER.solve(...; workspace=ws)`, the returned trajectory is copied by default so it remains valid after later solves reuse `ws`.  Set `copy_result=false` only when you are intentionally accepting workspace-owned output that may be overwritten by a later call.

---

## References

[^1]: Zoltowski, D. M., Wu, S., Gonzalez, X., Kozachkov, L., & Linderman, S. W. (2025). *Parallelizing MCMC Across the Sequence Length*. NeurIPS 2025. [arXiv:2508.18413](https://arxiv.org/abs/2508.18413)
