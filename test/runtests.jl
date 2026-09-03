using ParallelMCMC
using Test

#= Reactant tests are opt-in because Reactant_jll is a large download. Opting in
also requires Reactant in the test environment: it is only an [extra] in
test/Project.toml, which Pkg.test does not install. The Reactant-free checks of
the AutoReactant validation logic run unconditionally
(test-Reactant-Validation.jl). =#
const _RUN_REACTANT_TESTS =
    lowercase(get(ENV, "PARALLELMCMC_TEST_REACTANT", "false")) in ("1", "true", "yes")

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
            title = titlecase(replace(splitext(file[6:end])[1], "-" => " "))
            @testset verbose=true "$title" begin
                include(joinpath(root, file))  # robust if walkdir recurses
            end
        end
    end
end
