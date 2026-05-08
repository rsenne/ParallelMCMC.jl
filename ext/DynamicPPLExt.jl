module DynamicPPLExt

using ParallelMCMC
using ADTypes: ADTypes
using DynamicPPL: DynamicPPL
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using LogDensityProblems: LogDensityProblems
using ParallelMCMC.DEER: DEER

"""
    DensityModel(turing_model::DynamicPPL.Model; ad_backend=ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Duplicated), hvp=nothing)

Convenience constructor: wraps a DynamicPPL/Turing `@model` directly as a
`DensityModel`, automatically extracting parameter names and wiring up gradient
computation via DynamicPPL's `adtype` interface.

Requires `DynamicPPL` to be loaded.

# Example
```julia
using Turing, ParallelMCMC, MCMCChains

@model function mymodel(y)
    μ ~ Normal(0, 1)
    y ~ Normal(μ, 0.5)
end

model = DensityModel(mymodel(1.5))
chain = sample(model, AdaptiveMALASampler(0.3; n_warmup=500), 2_000;
               chain_type=MCMCChains.Chains, discard_warmup=true, progress=true)
```

# Notes
- Parameter names are extracted from the model's prior. For most common
  distributions (Normal, MvNormal, Exponential, etc.) the names match the
  unconstrained parameter space used by LogDensityProblems. If the extracted
  names do not match the dimensionality (e.g. due to simplex constraints), the
  constructor falls back to generic `x[1], x[2], ...` names with a warning.
"""
function ParallelMCMC.DensityModel(
    turing_model::DynamicPPL.Model;
    ad_backend=ADTypes.AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Duplicated,
    ),
    hvp=nothing,
)
    # Sample in linked/unconstrained space and let DynamicPPL provide the gradient.
    ld = DynamicPPL.LogDensityFunction(
        turing_model,
        DynamicPPL.getlogjoint_internal,
        DynamicPPL.LinkAll();
        adtype=ad_backend,
    )

    caps = LogDensityProblems.capabilities(ld)
    caps isa LogDensityProblems.LogDensityOrder{0} && error(
        "AD gradient setup failed. The wrapped model must support gradients. " *
        "Ensure your ad_backend is compatible.",
    )

    dim = LogDensityProblems.dimension(ld)

    # Try to extract parameter names; fall back to nothing on any error or mismatch.
    param_names = _try_extract_param_names(turing_model, dim)

    logp(x) = LogDensityProblems.logdensity(ld, x)
    function gradlogp(x)
        _, g = LogDensityProblems.logdensity_and_gradient(ld, x)
        return g
    end

    #=
    HVP backend deliberately chosen as ForwardDiff, not the package default
    forward-over-reverse Enzyme. Forward-over-reverse Enzyme on a DPPL
    `logdensity` crashes during compile with "broken gc calling conv fix" on
    MvNormal/Dirichlet (and likely other distributions whose logpdf boxes
    struct returns through Julia's sret ABI). Forward-over-forward and
    reverse-over-reverse Enzyme fail too; Mooncake can't trace the Enzyme
    `llvmcall` already used by the gradient. ForwardDiff over `logp` is the
    only configuration that produces a correct HVP for these models, and
    `logp` is a small scalar function so FD is the right cost regime anyway.
    =#
    if hvp === nothing
        hvp_backend = ADTypes.AutoForwardDiff()
        hvp_prep_ref = Ref{Any}(nothing)
        function _auto_hvp(x, v)
            if hvp_prep_ref[] === nothing
                hvp_prep_ref[] = DEER._prepare_logdensity_hvp(logp, hvp_backend, x)
            end
            return DEER._logdensity_hvp_prepared(logp, hvp_prep_ref[], hvp_backend, x, v)
        end
        hvp = _auto_hvp
    end

    return ParallelMCMC.DensityModel(logp, gradlogp, dim; hvp=hvp, param_names=param_names)
end

"""
Extract flat parameter names from a DynamicPPL model by sampling from the prior.
Returns a `Vector{Symbol}` if the count matches `expected_dim`, otherwise `nothing`.
"""
function _try_extract_param_names(model::DynamicPPL.Model, expected_dim::Int)
    try
        vi = DynamicPPL.VarInfo(model)
        names = Symbol[]
        for vn in keys(vi)
            val = vi[vn]
            sym = DynamicPPL.getsym(vn)
            if val isa Number
                push!(names, Symbol(sym))
            else
                for i in 1:length(val)
                    push!(names, Symbol("$(sym)[$i]"))
                end
            end
        end
        if length(names) == expected_dim
            return names
        else
            @warn "ParallelMCMC: parameter name extraction produced $(length(names)) names " *
                "but model has $expected_dim unconstrained dimensions " *
                "(likely due to bijector dimension changes, e.g. Dirichlet/LKJ constraints). " *
                "Falling back to generic x[1], x[2], ... names."
            return nothing
        end
    catch
        return nothing
    end
end

end # module
