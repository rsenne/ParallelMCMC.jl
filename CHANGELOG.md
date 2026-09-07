# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The derivative slots of `DensityModel` (`grad_logdensity`, `hvp`,
  `grad_logdensity_batch`, `hvp_batch`) now also accept
  `ADTypes.AbstractADType`. AD preparations are cached per model (#40, #52).
- `ParallelMALASampler`'s `backend` is optional when the model supplies its
  own HVPs (#52).
- Batched gradients are derived from `logdensity_batch` when
  `grad_logdensity` is a backend (#52).
- HVP backends over AD-derived gradients now use
  `DifferentiationInterface.SecondOrder`; explicit `SecondOrder` backends
  respect both configured passes (#37).
- `ADTypes.AutoReactant()` can supply derivatives and sampler HVPs through
  Reactant.jl. It requires `using Reactant`, a traceable log-density, and
  `AutoReactant()` on both AD passes of an HVP (#37, #52).
- The `DensityModel` constructors in `DynamicPPLExt` and `LogDensityProblemsExt`
  now accept `hvp` and the batched derivative slots.

### Fixed

- `SecondOrder` now respects both configured backends.
- Backend normalization no longer picks a differentiation mode on the user's
  behalf (#62). `AutoEnzyme()` keeps its unset mode. Pass
  `AutoEnzyme(; mode=Enzyme.Forward)` to request forward mode explicitly.
- Invalid `AutoReactant` backend combinations are rejected before compilation.

### Changed

- The AD-HVP fallback now respects the backend's `hvp_mode`, so
  `AutoEnzyme(mode=Enzyme.Reverse)` uses reverse-on-grad (#38).
- Deriving a batched gradient applies AD to `logdensity_batch`. Supply
  `grad_logdensity_batch` to avoid this.
- `ParallelMALASampler`'s `backend` no longer derives a batched gradient, only
  Hessian-vector products. Pass `grad_logdensity_batch` explicitly or use a
  backend in `grad_logdensity` to derive one.
- Batched derivative slots require `logdensity_batch`. An `hvp_batch` without
  a batched gradient now raises an error.

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
