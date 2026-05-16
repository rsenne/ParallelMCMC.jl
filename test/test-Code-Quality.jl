using Test
using Aqua, JET
using ParallelMCMC

@testset "Aqua" begin
    Aqua.test_all(ParallelMCMC)
end

@testset "JET" begin
    JET.test_package(ParallelMCMC; target_modules=(ParallelMCMC,))
end
