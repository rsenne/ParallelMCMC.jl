# API reference

```@meta
CurrentModule = ParallelMCMC
```

This page collects the public API and the low-level DEER types that are useful
for custom solvers. For an end-to-end example, begin with
[Getting started](10-getting-started.md).

## Model

```@docs
DensityModel
```

Additional constructors become available when their packages are loaded:

- `DensityModel(ld; param_names=nothing, kwargs...)` wraps a gradient-capable
  `LogDensityProblems` object.
- `DensityModel(turing_model; ad_backend, kwargs...)` wraps a DynamicPPL/Turing
  model, extracts its parameter names, and converts output back to the original
  parameter space.

See [Defining models](12-models.md) for examples and the derivative-slot rules.

## Samplers

```@docs
MALASampler
AdaptiveMALASampler
ParallelMALASampler
```

All three implement the AbstractMCMC interface. The usual entry point is
`sample(model, sampler, n; kwargs...)`, optionally with an RNG or an
AbstractMCMC ensemble such as `MCMCThreads()`.

## Transition and state types

These types are part of the AbstractMCMC protocol. Most users receive them from
`AbstractMCMC.step` and do not construct them directly.

```@docs
MALAState
MALATransition
AdaptiveMALAState
AdaptiveMALATransition
ParallelMALAState
ParallelMALATransition
MALATapeElement
```

## Device support

Loading CUDA activates the package extension for `CuArray` storage. Loading
Reactant activates `ADTypes.AutoReactant()` support for derivative slots and
the parallel sampler's `backend`. The restrictions and backend comparison are
documented in [GPU execution](15-gpu.md).

```@docs
ParallelMCMC.needs_host_staging
```

## Low-level DEER API

[`ParallelMALASampler`](@ref) builds on the types below. Use them directly when
solving a custom taped recursion or managing workspaces across repeated solves.

```@docs
DEER.TapedRecursion
DEER.DEERWorkspace
DEER.solve
ParallelMCMC.DEERScan.AffineScanWorkspace
MALA.mala_step_surrogate_sigmoid
```

The in-place diagonal scan is available as
`ParallelMCMC.DEERScan.solve_affine_scan_diag!`. See
[How the algorithm works](20-algorithms.md) before using the low-level API.

## Index

```@index
Pages = ["95-reference.md"]
```
