function _construct_flexichain(
    ::Type{TKey},
    vals::AbstractMatrix{<:Real},
    internals::NamedTuple,
    param_names::Any,
    model::DensityModel,
) where {TKey}
    # Wrap user-supplied names in `Parameter`. This allows people to specify, e.g.,
    # `param_names=(:x, :y, :z=>(2,))` without faffing with `Parameter` themselves. Also
    # 'upgrade' symbol parameter names to VarNames if the user requested a VNChain.
    to_parameter(vn::VarName) = FlexiChains.Parameter(vn)
    to_parameter(s::Symbol) = FlexiChains.Parameter(TKey <: VarName ? VarName{s}() : s)

    N, D = size(vals)

    if param_names === nothing
        param_names = model.param_names
    end
    wrapped_param_names = if param_names === nothing
        # Wasn't defined either as `param_names` or `model.param_names`, so make some up
        (to_parameter(:x) => (D,),)
    else
        map(param_names) do n
            if n isa Pair
                to_parameter(n.first) => n.second
            elseif n isa TKey || n isa Symbol
                to_parameter(n)
            else
                throw(
                    ArgumentError(
                        "param_names must be a collection of Pairs, Symbols, or $TKey, got $(typeof(n))",
                    ),
                )
            end
        end
    end

    arr = reshape(vals, N, 1, D)
    param_chain = FlexiChains.FlexiChain{TKey}(arr, Tuple(wrapped_param_names))

    isempty(internals) && return param_chain

    #=
    The internals have mixed element types (e.g. `Bool` for `accepted`/`is_warmup`,
    `FP` for `logp`), so they cannot share a single stacked array without widening.
    Store each in its own column via the dict-of-arrays constructor, which preserves
    the natural element type (see issue #49), then merge with the parameters.
    =#
    extras = OrderedDict{FlexiChains.ParameterOrExtra{<:TKey},AbstractArray}(
        FlexiChains.Extra(name) => internals[name] for name in keys(internals)
    )
    extra_chain = FlexiChains.FlexiChain{TKey}(N, 1, extras)

    return merge(param_chain, extra_chain)
end
