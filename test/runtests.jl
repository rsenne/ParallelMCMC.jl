using ParallelMCMC
using Test

#= `Reactant_jll` ships a prebuilt XLA and is a large download, so `Reactant` sits
in test/Project.toml's `[extras]` and `test-Reactant-HVP.jl` runs only when this
is set. Add Reactant to the test environment as well; the file's own
`try ... using Reactant ... catch` skips its testsets if it still won't load. =#
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
