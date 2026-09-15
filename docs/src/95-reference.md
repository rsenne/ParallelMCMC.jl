# [API Reference](@id reference)

```@meta
CurrentModule = ParallelMCMC
```

This page documents all public types and functions exported by ParallelMCMC.jl.

## Reactant

Loading `Reactant` (`using Reactant`) enables `ADTypes.AutoReactant()` as a
backend for derivatives in `DensityModel` and as `ParallelMALASampler`'s
`backend`. See [GPU Execution](15-gpu.md) for its constraints.

## CUDA and other device arrays

`using CUDA` loads the `CUDAExt` extension; see [GPU Execution](15-gpu.md).

```@docs
ParallelMCMC.needs_host_staging
```

## Extension constructor

`DensityModel(ld; param_names=nothing)` wraps any `LogDensityProblems` object with gradients when `LogDensityProblems` is loaded. See [LogDensityProblems models](@ref) for an example.

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

Use these functions and workspaces to solve taped recursions or run the
diagonal affine scan directly.

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
