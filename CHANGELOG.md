# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2]

### Added

- DynamicPPL-backed `DensityModel`s now map samples back to the original
  (possibly constrained) parameter space, with names taken directly from the
  Turing model. This works correctly for distributions whose dimension changes
  under linking (e.g. `Dirichlet`, `LKJ`, `product_distribution` with
  `NamedTuple` keys).
- Chains from Turing models now expose `:logjoint`, `:logprior`, and
  `:loglikelihood` as separate columns.
- `hvp` keyword argument is now forwarded through the `LogDensityProblems`-based
  `DensityModel` constructor.

### Changed

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

- Heuristic prior-based parameter-name extraction (`_try_extract_param_names`)
  and its warning fallback for dimension-changing bijectors.

## [0.0.1]

- Initial release.
