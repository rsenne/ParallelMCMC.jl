# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
