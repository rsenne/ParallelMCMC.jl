# Troubleshooting

## The DEER solve takes many iterations

First reduce `epsilon` or `T`. If the model otherwise behaves well, lower
`damping` from `0.5` toward `0.3`. Do not loosen the convergence tolerances
until you have ruled out a derivative error.

## Parallel MALA is slower than sequential MALA

That is expected for small, cheap targets on a CPU. Parallel MALA has setup,
derivative, and scan overhead, and is designed for targets with enough batched
work to benefit from parallel hardware.

When comparing the two, benchmark the complete sampling call, and measure
compilation separately from steady-state execution.

## The chain does not move or has many rejections

Run [`MALASampler`](@ref) or [`AdaptiveMALASampler`](@ref) first. Check the
gradient against finite differences or an independent AD backend, then try a
smaller step size.

## GPU code reports scalar indexing or AD compilation errors

The model must be written in GPU-compatible array operations.

AD backend support also differs on the GPU. See [GPU execution](15-gpu.md) for
the current limitations.
