# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The derivative slots of `DensityModel` (`grad_logdensity`, `hvp`,
  `grad_logdensity_batch`, `hvp_batch`) now take an `ADTypes.AbstractADType`
  in place of a callable, so `DensityModel(logp, AutoForwardDiff(), dim)`
  builds a model from the log-density alone. Backends are turned into
  prepared DifferentiationInterface callables when sampling starts, and that
  preparation is reused across steps (#40, #52). The prepared model rides
  along in the sampler state to get that reuse; a state handed to `step` for a
  different model, which `initial_state` allows, is re-prepared from the model
  passed rather than reused, so the model given to `sample` is the one sampled.
- `ParallelMALASampler`'s `backend` keyword is now optional. It is only the
  fallback source of Hessian-vector products, so a `DensityModel` carrying
  its own `hvp` / `hvp_batch` does not need it (#52). With no sampler
  `backend`, a batched HVP is derived from the model's own `hvp` backend.
- A `logdensity_batch` given without a `grad_logdensity_batch` now has the
  batched gradient derived for it when `grad_logdensity` is a backend, rather
  than leaving the batched DEER path switched off (#52). `hvp_batch` can be a
  backend in that case too, and differentiates the derived gradient.
- An HVP backend over an AD-derived gradient is now taken as true second-order
  AD, `DifferentiationInterface.SecondOrder(hvp_backend, grad_backend)` handed
  to `DI.hvp`, instead of an outer AD pass over the prepared DI gradient (#37).
  That nesting dropped out of its preparation as soon as the outer pass pushed
  tangents in, so the composed operator is both what was asked for and cheaper:
  around 10x fewer allocations for a `logdensity`-only model. A backend over a
  hand-written gradient still differentiates that gradient once, as before.
- `hvp` / `hvp_batch` accept a `SecondOrder` with both halves honoured, meaning
  the log-density is differentiated twice and the gradient slot is not the inner
  pass. Previously the inner half was silently discarded and only the outer used.
  This is the one AD route to an HVP for a Turing or LogDensityProblems model,
  whose gradient arrives already prepared and so cannot be differentiated again.
- The `DensityModel` constructors in `DynamicPPLExt` and `LogDensityProblemsExt`
  forward `logdensity_batch`, `grad_logdensity_batch` and `hvp_batch`, so a
  Turing or LogDensityProblems model can reach the batched DEER path. Neither
  provides a batched log-density, so `logdensity_batch` has to be written by hand.
- Adds `JuliaFormatter` testing which was forgotten (#60).

### Fixed

- The reverse-on-grad HVP path differentiated with `DI.inner(backend)` while
  its strategy was routed on `DI.outer(backend)`, so a
  `DifferentiationInterface.SecondOrder` ran the wrong half of the pair. A
  `SecondOrder` now goes to the true second-order path instead of either
  strategy, and the half-selecting helper it still uses agrees: normalization
  applies to the outer pass. Unwrapping to that half happens before the
  normalization hook is dispatched on, so a `SecondOrder(AutoEnzyme(), ...)` still
  reaches `EnzymeExt` and gets its function annotation filled in rather than
  running as a bare `AutoEnzyme()`.
- Backend normalization no longer picks a differentiation mode on the user's
  behalf (#62). `EnzymeExt` pinned `mode=Enzyme.Forward` (with
  `set_runtime_activity`) onto an `AutoEnzyme()` left mode-agnostic, on the
  grounds that reverse mode hit a gc-transition abort on GPU and that composed
  `pmcmc_matmul` calls needed runtime activity. The `pmcmc_*` Enzyme rules keep
  Enzyme off both paths on their own now, so the pin bought nothing — and it cost
  correctness, because it silently rewrote the direction of a `SecondOrder`'s
  outer half. `SecondOrder(AutoEnzyme(), AutoForwardDiff())` is
  reverse-over-forward to `hvp_mode`, its inner half being forward-only, and came
  out forward-over-forward. Normalization now fills in only
  `function_annotation=Enzyme.Const`, which is about this package's own read-only
  HVP wrappers rather than about Enzyme's mode, and leaves `mode` exactly as given
  — unset included, for DI to resolve from the operator it runs. `hvp_mode` is
  therefore identical before and after normalization for every backend pair.

  A mode set explicitly was never overridden, so only mode-agnostic backends were
  affected, and the HVP was a correct HVP either way; what changes is that the
  composition asked for is the one that runs. The two normalization hooks
  (`_hvp_forward_backend`, `_hvp_closure_backend`) collapse into a single
  `_normalized_backend`, since without a mode to choose they no longer differ.
  Users relying on a plain `AutoEnzyme()` being run forward should now pass
  `AutoEnzyme(; mode=Enzyme.Forward)` explicitly.

### Changed

- The AD-HVP fallback strategy (forward-on-grad vs reverse-on-grad) now comes
  from DifferentiationInterface's `hvp_mode` trait rather than a hardcoded
  per-backend list, so `AutoEnzyme(mode=Enzyme.Reverse)` routes to the
  reverse-on-grad path (#38).
- Because a `logdensity_batch` without a `grad_logdensity_batch` now has the
  batched gradient derived rather than switching the batched DEER path off, a
  model whose `grad_logdensity` is a backend runs the batched update where it
  used to run the unbatched one, and AD is applied to its `logdensity_batch`. On
  GPU that subjects a function nothing was differentiating before to the
  backend's restrictions (`pmcmc_*` wrappers for Enzyme). Supply
  `grad_logdensity_batch` to keep AD out of the batched path. A model with a
  hand-written `grad_logdensity` is unaffected: nothing derives a batched
  gradient for it, so the batched path stays off as before.
- `ParallelMALASampler`'s `backend` no longer derives a batched gradient, only
  Hessian-vector products. It could previously switch the batched DEER path on
  for a model with a hand-written gradient, which made a keyword that reads as
  an HVP fallback decide which update path ran and put AD on a
  `logdensity_batch` the user had not opted into differentiating. Models that
  relied on that should pass `grad_logdensity_batch` explicitly, or a backend in
  `grad_logdensity` for one to be derived from.
- Both batched derivative slots now require `logdensity_batch`, which the
  batched update evaluates directly, and the constructor rejects them without
  one. A callable `grad_logdensity_batch` or `hvp_batch` supplied on its own
  used to be accepted and then silently ignored. `logdensity_batch` alone is
  still valid and still used to score whole trajectories.
- An `hvp_batch` that reaches sampling with no batched gradient to pair it with
  now raises rather than silently falling back to the unbatched update.

### Removed

- `DynamicPPLExt` no longer requires `ForwardDiff` as a triggering library to load.

## [0.2.0] - 2026-06-29

### Changed

- **FlexiChains is now the default (and only built-in) chain type.**
  `sample(model, sampler, N; chain_type=...)` returns a
  `FlexiChains.FlexiChain` instead of an `MCMCChains.Chains`. Use
  `chain_type=SymChain` for `Symbol`-keyed chains or `chain_type=VNChain` for
  `VarName`-keyed chains (#44).
- DynamicPPL-backed models must use `VNChain`; requesting `SymChain` for a
  DynamicPPL model now throws an `ArgumentError` (#44).
- `param_names` handling is more forgiving: user-supplied names are wrapped in
  `FlexiChains.Parameter` automatically, `Symbol` names are upgraded to
  `VarName`s when a `VNChain` is requested, and better default names are
  generated when none are supplied (#44).
- `DynamicPPL` compat bumped to `0.41.6, 0.42` (#44).

### Added

- `FlexiChains` dependency; `DynamicPPLExt` now also loads on `FlexiChains`
  (#44).

### Removed

- `MCMCChains` dependency. Chains are now built entirely on FlexiChains; there
  is no longer any MCMCChains output (#44).

## [0.1.0] - 2026-06-17

### Added

- GPU support for DEER-based parallel sampling, including a dedicated guide
  (`docs/src/15-gpu.md`) and a Bayesian logistic-regression GPU example.
- `EnzymeExt` extension providing backend-specific reverse-mode Enzyme rules
  for the `pmcmc_*` wrappers, which work around the CUDA gc-transition abort
  Enzyme hits when differentiating matmul/reductions on GPU.
- Exported `pmcmc_matmul`, `pmcmc_dot`, and `pmcmc_dotsum` — `Base`-equivalent
  wrappers with stable function identities so the `EnzymeExt` AD rules fire.
  Use these in model code that must run Enzyme on GPU.
- Per-backend HVP strategy dispatch (`ForwardOnGrad` / `ReverseOnGrad`):
  forward-on-gradient for forward-capable backends (`AutoEnzyme`,
  `AutoForwardDiff`, forward Mooncake) and reverse-on-gradient routing for
  `AutoMooncake`, `AutoZygote`, `AutoReverseDiff`, and `AutoTracker`.
- DynamicPPL-backed `DensityModel`s now map samples back to the original
  (possibly constrained) parameter space, with names taken directly from the
  Turing model. This works correctly for distributions whose dimension changes
  under linking (e.g. `Dirichlet`, `LKJ`, `product_distribution` with
  `NamedTuple` keys).
- Chains from Turing models now expose `:logjoint`, `:logprior`, and
  `:loglikelihood` as separate columns.
- `hvp` keyword argument is now forwarded through the `LogDensityProblems`-based
  `DensityModel` constructor.
- Test coverage for GPU AD-HVP, GPU MALA, GPU performance, owned-matmul AD
  rules, and a Zygote backend.

### Changed

- `DifferentiationInterface` is now the single user-facing entry point into AD.
  Backends are passed as `ADTypes` AD types and HVP routing is resolved
  statically via singleton-trait dispatch, making the AD-HVP fallback
  type-stable.
- Enzyme is now an optional dependency loaded through `EnzymeExt` rather than a
  hard dependency; `Enzyme` compat bumped to `0.13.146`.
- `DynamicPPL` compat bumped to `0.40.6, 0.41`.
- Parameter names for Turing models are now derived by reevaluating the model
  via `DynamicPPL.ParamsWithStats` rather than extracted heuristically at
  construction time. The `param_names` field on `DensityModel` is no longer
  populated by the DynamicPPL convenience constructor.
- `model.logdensity` and `model.grad_logdensity` constructed from a
  `LogDensityProblems` object are now `LogDensityProblemPrimal` /
  `LogDensityProblemGradient` callable structs rather than anonymous closures.
  Calling behaviour is unchanged; only the concrete type differs.

### Removed

- `DEER.DEFAULT_BACKEND` / `DEFAULT_HVP_BACKEND` and the old default-Enzyme
  machinery. **`backend` is now a required keyword argument on
  `ParallelMALASampler`** — there is no implicit default. To reproduce the
  previous behaviour, load `Enzyme` and pass
  `backend=AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Duplicated)`.
- Heuristic prior-based parameter-name extraction (`_try_extract_param_names`)
  and its warning fallback for dimension-changing bijectors.

## [0.0.1]

- Initial release.
