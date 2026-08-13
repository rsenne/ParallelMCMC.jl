# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The derivative slots of `DensityModel` (`grad_logdensity`, `hvp`,
  `grad_logdensity_batch`, `hvp_batch`) now accept an `ADTypes.AbstractADType`
  in place of a callable, so `DensityModel(logp, AutoForwardDiff(), dim)` builds
  a model from the log-density alone (#40, #52). Backends become prepared
  DifferentiationInterface callables when sampling starts, and the prepared
  model rides along in the sampler state so the preparation is reused. A state
  handed to `step` for a different model — which `initial_state` allows — is
  re-prepared, so the model given to `sample` is the one sampled.
- `ParallelMALASampler`'s `backend` keyword is now optional: it is only the
  fallback source of Hessian-vector products, so a `DensityModel` carrying its
  own `hvp` / `hvp_batch` does not need one. Without it, a batched HVP comes
  from the model's own `hvp` backend (#52).
- A `logdensity_batch` given without a `grad_logdensity_batch` now has the
  batched gradient derived from it when `grad_logdensity` is a backend, instead
  of leaving the batched DEER path switched off. `hvp_batch` can be a backend in
  that case too, and differentiates the derived gradient (#52).
- An HVP backend over an AD-derived gradient is now true second-order AD,
  `DifferentiationInterface.SecondOrder(hvp_backend, grad_backend)` handed to
  `DI.hvp`, rather than an outer AD pass over the prepared DI gradient — which
  dropped out of its preparation as soon as tangents were pushed through it
  (#37). Around 10x fewer allocations for a `logdensity`-only model. A backend
  over a hand-written gradient still differentiates that gradient once.
- `hvp` / `hvp_batch` accept a `SecondOrder` with both halves honoured, so the
  log-density is differentiated twice and the gradient slot is not the inner
  pass. The inner half used to be discarded. This is the one AD route to an HVP
  for a Turing or LogDensityProblems model, whose gradient arrives already
  prepared and cannot be differentiated again.
- New `ReactantExt`. `ADTypes.AutoReactant()` in a derivative slot, or as the
  sampler `backend`, traces the derivative with Enzyme-MLIR and compiles it to
  an XLA executable via Reactant.jl, off Enzyme's LLVM pipeline and off
  DifferentiationInterface entirely, which yields a true second-order HVP for
  a log-density-only model (#37, #52). Requires `using Reactant` and a
  Reactant-traceable log-density. `AutoReactant` does not pair with a
  DifferentiationInterface backend, a `LogDensityProblems` gradient, or a
  `SecondOrder` across the two passes of an HVP, and doing so raises an
  `ArgumentError` before any AD runs, as does a non-default
  `AutoReactant(; mode=...)`. Two caveats, spelled out in `ext/ReactantExt.jl`'s
  module docstring and the Reactant section of `docs/src/15-gpu.md`: a traced
  log-density must be pure with respect to the data it captures, since a captured
  array mutated after preparation stays frozen at its old value in every
  derivative compiled from it, and the compiled derivative runs on whatever
  device Reactant's XLA client targets, which need not be the GPU the model's
  arrays live on.
- The `DensityModel` constructors in `DynamicPPLExt` and `LogDensityProblemsExt`
  forward `logdensity_batch`, `grad_logdensity_batch` and `hvp_batch`, so a
  Turing or LogDensityProblems model can reach the batched DEER path. Neither
  provides a batched log-density, so `logdensity_batch` has to be written by hand.
- Adds `JuliaFormatter` testing which was forgotten (#60).

### Changed

- The AD-HVP fallback strategy (forward-on-grad vs reverse-on-grad) now comes
  from DifferentiationInterface's `hvp_mode` trait rather than a hardcoded
  per-backend list, so `AutoEnzyme(mode=Enzyme.Reverse)` routes to the
  reverse-on-grad path (#38).
- A model whose `grad_logdensity` is a backend now runs the batched update where
  it used to run the unbatched one, since a `logdensity_batch` without a
  `grad_logdensity_batch` has one derived for it. On GPU that puts the backend's
  restrictions (`pmcmc_*` wrappers for Enzyme) on a `logdensity_batch` nothing
  was differentiating before; supply `grad_logdensity_batch` to keep AD out of
  it. A hand-written `grad_logdensity` is unaffected.
- `ParallelMALASampler`'s `backend` no longer derives a batched gradient, only
  Hessian-vector products. It could previously switch the batched DEER path on
  for a model with a hand-written gradient, which let a keyword that reads as an
  HVP fallback decide which update path ran. Pass `grad_logdensity_batch`
  explicitly, or a backend in `grad_logdensity` to derive one from.
- Both batched derivative slots now require `logdensity_batch`, which the batched
  update evaluates directly, and the constructor rejects them without one. A
  callable `grad_logdensity_batch` or `hvp_batch` on its own used to be accepted
  and then ignored. `logdensity_batch` alone still scores whole trajectories.
- An `hvp_batch` that reaches sampling with no batched gradient to pair it with
  now raises rather than falling back to the unbatched update.

### Fixed

- The reverse-on-grad HVP path differentiated with `DI.inner(backend)` while its
  strategy was routed on `DI.outer(backend)`, so a
  `DifferentiationInterface.SecondOrder` ran the wrong half of the pair. A
  `SecondOrder` now goes to the second-order path instead of either strategy, and
  normalization applies to the outer half after unwrapping, so
  `SecondOrder(AutoEnzyme(), ...)` still reaches `EnzymeExt` rather than running
  as a bare `AutoEnzyme()`.
- Backend normalization no longer picks a differentiation mode on the user's
  behalf (#62). `EnzymeExt` pinned `mode=Enzyme.Forward`, with
  `set_runtime_activity`, onto a mode-agnostic `AutoEnzyme()`, against a
  gc-transition abort on GPU and an `EnzymeRuntimeActivityError` on composed
  `pmcmc_matmul` calls. The `pmcmc_*` Enzyme rules keep Enzyme off both paths on
  their own, so the pin bought nothing and cost correctness: it rewrote the
  direction of a `SecondOrder`'s outer half, turning
  `SecondOrder(AutoEnzyme(), AutoForwardDiff())` — reverse-over-forward to
  `hvp_mode`, its inner half being forward-only — into forward-over-forward.
  Normalization now fills in `function_annotation=Enzyme.Const` and leaves `mode`
  exactly as given, unset included, so `hvp_mode` reads the same before and after
  it for every pair. Only mode-agnostic backends were affected and the HVP was
  correct either way; what changes is that the composition asked for is the one
  that runs. The two hooks `_hvp_forward_backend` and `_hvp_closure_backend`
  collapse into one `_normalized_backend`. Pass
  `AutoEnzyme(; mode=Enzyme.Forward)` to keep the old direction.
- `_check_reactant_pair` now runs before `_prepare_model` resolves a gradient, so
  a mismatched `AutoReactant` pairing is reported immediately instead of after a
  full XLA compile, and `AutoReactant` nested inside a `SecondOrder` is rejected
  there too rather than reaching `DI.prepare_hvp`. An `AutoReactant` gradient
  composes through the same second-order branch as any other AD-derived gradient,
  and `_hvp_strategy(::AutoReactant)` is now live rather than a dead branch.
- `test/test-Reactant-HVP.jl` asserts a posterior mean against a target with a
  known mean, and cross-checks against the analytic HVP on the same noise tape.
  `size(chain)` and `all(isfinite, ...)` pass even for a badly wrong HVP, since
  DEER's Newton iteration then just fails to converge rather than producing
  `NaN`s.

### Removed

- `DynamicPPLExt` no longer requires `ForwardDiff` as a triggering library to load.
- `Reactant` moved from `test/Project.toml`'s `[deps]` to `[extras]`, and
  `test/test-Reactant-HVP.jl` is skipped unless `PARALLELMCMC_TEST_REACTANT` is
  set. `Reactant_jll` ships a prebuilt XLA and is a large download most CI runs
  and most local `Pkg.test()` calls should not have to pay for. Opt in by setting
  the env var and adding `Reactant` to the test environment.

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
