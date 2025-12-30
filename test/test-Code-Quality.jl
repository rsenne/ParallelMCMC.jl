using Aqua, JET

@testset "Code Quality" begin
    @testset "Aqua Tests" begin
        Aqua.test_all(ParallelMCMC)
    end
    
    @testset "Code Linting (JET)" begin
        JET.test_package(ParallelMCMC; target_modules=(ParallelMCMC,))
    end
end