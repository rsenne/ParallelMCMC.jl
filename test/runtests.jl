using ParallelMCMC
using Test

#= Reactant tests are opt-in because Reactant_jll is a large download. Opting in
also requires Reactant in the test environment: it is only an [extra] in
test/Project.toml, which Pkg.test does not install. The Reactant-free checks of
the AutoReactant validation logic run unconditionally
(test-Reactant-Validation.jl). =#
const _RUN_REACTANT_TESTS =
    lowercase(get(ENV, "PARALLELMCMC_TEST_REACTANT", "false")) in ("1", "true", "yes")

#= Wren tests are opt-in because WrenCore and WrenDists are not registered. Opting
in also requires both in the test environment, e.g. via `Pkg.develop(path=...)`.
They are listed as [extras] in test/Project.toml only to record their UUIDs. =#
const _RUN_WREN_TESTS =
    lowercase(get(ENV, "PARALLELMCMC_TEST_WREN", "false")) in ("1", "true", "yes")

@testset verbose=true "ParallelMCMC" begin
    #=
    Don't add your tests to runtests.jl. Instead, create files named

        test-title-for-my-test.jl

    The file will be automatically included inside a `@testset` with title "Title For My Test".
    =#
    for (root, dirs, files) in walkdir(@__DIR__)
        for file in files
            if isnothing(match(r"^test-.*\.jl$", file))
                continue
            end
            if file == "test-Reactant-HVP.jl" && !_RUN_REACTANT_TESTS
                @info "Skipping $file (set PARALLELMCMC_TEST_REACTANT=true to opt in)"
                continue
            end
            if file == "test-Wren-Extension.jl" && !_RUN_WREN_TESTS
                @info "Skipping $file (set PARALLELMCMC_TEST_WREN=true to opt in)"
                continue
            end
            title = titlecase(replace(splitext(file[6:end])[1], "-" => " "))
            @testset verbose=true "$title" begin
                include(joinpath(root, file))  # robust if walkdir recurses
            end
        end
    end
end
