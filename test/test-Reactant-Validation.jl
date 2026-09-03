using Test
using ParallelMCMC
using ADTypes
using ForwardDiff: ForwardDiff
using LogDensityProblems: LogDensityProblems

#= The AutoReactant validation in src/interface.jl needs no Reactant, so it runs
in every test invocation. The Reactant-backed tests are opt-in in
test-Reactant-HVP.jl. =#

const DI_RV = ParallelMCMC.DEER.DI
const D_RV = 4

logp_rv(x) = -0.25 * sum(abs2, x)^2
gradlogp_rv(x) = -sum(abs2, x) .* x

struct _FakeLD_RV end
LogDensityProblems.capabilities(::_FakeLD_RV) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(::_FakeLD_RV) = D_RV
function LogDensityProblems.logdensity_and_gradient(::_FakeLD_RV, x)
    return logp_rv(x), gradlogp_rv(x)
end

@testset "Reactant does not pair with a DI backend" begin
    @test ParallelMCMC._check_reactant_pair(AutoReactant(), AutoReactant()) === nothing
    @test ParallelMCMC._check_reactant_pair(nothing, AutoReactant()) === nothing
    @test ParallelMCMC._check_reactant_pair(AutoForwardDiff(), AutoForwardDiff()) ===
        nothing

    @test_throws ArgumentError ParallelMCMC._check_reactant_pair(
        AutoReactant(), AutoForwardDiff()
    )
    @test_throws ArgumentError ParallelMCMC._check_reactant_pair(
        AutoForwardDiff(), AutoReactant()
    )
end

@testset "AutoReactant cannot appear inside a SecondOrder" begin
    @test ParallelMCMC._check_reactant_pair(
        nothing, DI_RV.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    ) === nothing

    @test_throws ArgumentError ParallelMCMC._check_reactant_pair(
        nothing, DI_RV.SecondOrder(AutoReactant(), AutoReactant())
    )
    @test_throws ArgumentError ParallelMCMC._check_reactant_pair(
        AutoForwardDiff(), DI_RV.SecondOrder(AutoReactant(), AutoForwardDiff())
    )
    @test_throws ArgumentError ParallelMCMC._check_reactant_pair(
        AutoReactant(), DI_RV.SecondOrder(AutoReactant(), AutoReactant())
    )
end

@testset "a LogDensityProblems gradient cannot pair with an AutoReactant hvp" begin
    @test ParallelMCMC._check_reactant_hvp_source(
        ParallelMCMC.LogDensityProblemGradient(nothing), AutoForwardDiff()
    ) === nothing
    @test_throws ArgumentError ParallelMCMC._check_reactant_hvp_source(
        ParallelMCMC.LogDensityProblemGradient(nothing), AutoReactant()
    )

    model = DensityModel(_FakeLD_RV(); hvp=AutoReactant())
    @test_throws ArgumentError ParallelMCMC._prepare_model(model, zeros(D_RV), 8, nothing)
end

@testset "mixed pairs are refused at preparation" begin
    x = zeros(D_RV)

    reactant_grad = DensityModel(logp_rv, AutoReactant(), D_RV; hvp=AutoForwardDiff())
    @test_throws ArgumentError ParallelMCMC._prepare_model(reactant_grad, x, 8, nothing)

    reactant_hvp = DensityModel(logp_rv, AutoForwardDiff(), D_RV; hvp=AutoReactant())
    @test_throws ArgumentError ParallelMCMC._prepare_model(reactant_hvp, x, 8, nothing)
end

# The extension shadows these fallbacks once Reactant is loaded, so the check
# only applies while it is absent.
if Base.get_extension(ParallelMCMC, :ReactantExt) === nothing
    @testset "clear load-hint error without Reactant loaded" begin
        model_grad = DensityModel(logp_rv, AutoReactant(), D_RV)
        @test_throws "AutoReactant requires Reactant.jl" ParallelMCMC._prepare_model(
            model_grad, zeros(D_RV), 8, AutoReactant()
        )

        model_hvp = DensityModel(logp_rv, gradlogp_rv, D_RV; hvp=AutoReactant())
        @test_throws "AutoReactant requires Reactant.jl" ParallelMCMC._prepare_model(
            model_hvp, zeros(D_RV), 8, nothing
        )
    end
end
