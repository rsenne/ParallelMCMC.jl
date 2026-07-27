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
  batched gradient derived for it, from the gradient slot's backend if it has
  one and the sampler's otherwise, rather than leaving the batched DEER path
  switched off (#52). `hvp_batch` can be a backend in that case too, and
  differentiates the derived gradient; with no backend anywhere to derive from,
  it raises when sampling starts.
- Adds `JuliaFormatter` testing which was forgotten (#60).

### Fixed

- The reverse-on-grad HVP path differentiated with `DI.inner(backend)` while
  its strategy was routed on `DI.outer(backend)`, so an `hvp` or `backend`
  given as a `DifferentiationInterface.SecondOrder` ran the wrong half of the
  pair. Both paths now take the outer, which is the pass being run--the
  gradient slot is the inner one. Unwrapping to the outer half happens before
  the backend-specific normalization hooks are dispatched on, so a
  `SecondOrder(AutoEnzyme(), ...)` still reaches `EnzymeExt` and gets its mode
  and function annotation pinned rather than running as a bare `AutoEnzyme()`
  (which aborts on GPU).

### Changed

- The AD-HVP fallback strategy (forward-on-grad vs reverse-on-grad) now comes
  from DifferentiationInterface's `hvp_mode` trait rather than a hardcoded
  per-backend list, so `AutoEnzyme(mode=Enzyme.Reverse)` routes to the
  reverse-on-grad path (#38).
- Because a `logdensity_batch` without a `grad_logdensity_batch` now has the
  batched gradient derived rather than switching the batched DEER path off, a
  model in that shape runs the batched update where it used to run the unbatched
  one, and AD is applied to its `logdensity_batch`. On GPU that subjects a
  function nothing was differentiating before to the backend's restrictions
  (`pmcmc_*` wrappers for Enzyme). Supply `grad_logdensity_batch` to keep AD out
  of the batched path.

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
