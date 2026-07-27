using Test
using Aqua, JET, JuliaFormatter
using ParallelMCMC

@testset "Blue Formatting" begin
    @test JuliaFormatter.format(ParallelMCMC; verbose=false, overwrite=false)
end

@testset "Aqua" begin
    Aqua.test_all(ParallelMCMC)
end

@testset "JET" begin
    JET.test_package(ParallelMCMC; target_modules=(ParallelMCMC,))
end
