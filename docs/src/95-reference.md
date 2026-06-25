# [API Reference](@id reference)

```@meta
CurrentModule = ParallelMCMC
```

This page documents all public types and functions exported by ParallelMCMC.jl.

## Extension constructors

`DensityModel` also has extension constructors for common probabilistic-programming interfaces:

- `DensityModel(ld; param_names=nothing)` for `LogDensityProblems` models with gradients
- `DensityModel(turing_model)` for `DynamicPPL` / Turing models when the relevant extension packages are loaded

See [Getting Started](10-getting-started.md) for end-to-end examples of both.

## Model

```@docs
DensityModel
```

## Samplers

```@docs
MALASampler
AdaptiveMALASampler
ParallelMALASampler
```

## Internal types

These types appear in the `AbstractMCMC` state/transition protocol.  You generally do not need to construct them directly.

```@docs
MALATapeElement
MALAState
MALATransition
AdaptiveMALAState
AdaptiveMALATransition
ParallelMALAState
ParallelMALATransition
```

## Low-level namespaces

These lower-level building blocks power the public samplers and are useful if you
want to work with taped recursions or the diagonal affine scan directly.

```@docs
DEER.TapedRecursion
DEER.DEERWorkspace
DEER.solve
ParallelMCMC.DEERScan.AffineScanWorkspace
MALA.mala_step_surrogate_sigmoid
```

The diagonal scan implementation itself lives in `ParallelMCMC.DEERScan.solve_affine_scan_diag!`.

## Index

```@index
Pages = ["95-reference.md"]
```
