using Test
using ADTypes
using Enzyme: Enzyme
using ParallelMCMC

const DEER_STRAT = ParallelMCMC.DEER
const DI_STRAT = ParallelMCMC.DEER.DI

@testset "HVP strategy from DI.hvp_mode" begin
    @testset "forward-capable backends → ForwardOnGrad" begin
        @test DEER_STRAT._hvp_strategy(AutoForwardDiff()) isa DEER_STRAT.ForwardOnGrad
        @test DEER_STRAT._hvp_strategy(AutoEnzyme()) isa DEER_STRAT.ForwardOnGrad
        @test DEER_STRAT._hvp_strategy(AutoEnzyme(; mode=Enzyme.Forward)) isa
            DEER_STRAT.ForwardOnGrad
        @test DEER_STRAT._hvp_strategy(AutoMooncakeForward()) isa DEER_STRAT.ForwardOnGrad
    end

    @testset "reverse-only backends → ReverseOnGrad" begin
        @test DEER_STRAT._hvp_strategy(AutoMooncake()) isa DEER_STRAT.ReverseOnGrad
        @test DEER_STRAT._hvp_strategy(AutoZygote()) isa DEER_STRAT.ReverseOnGrad
        @test DEER_STRAT._hvp_strategy(AutoReverseDiff()) isa DEER_STRAT.ReverseOnGrad
        @test DEER_STRAT._hvp_strategy(AutoTracker()) isa DEER_STRAT.ReverseOnGrad
        @test DEER_STRAT._hvp_strategy(AutoEnzyme(; mode=Enzyme.Reverse)) isa
            DEER_STRAT.ReverseOnGrad
    end

    @testset "SecondOrder follows hvp_mode's composition" begin
        so_fwd_outer = DI_STRAT.SecondOrder(AutoForwardDiff(), AutoZygote())
        @test DEER_STRAT._hvp_strategy(so_fwd_outer) isa DEER_STRAT.ForwardOnGrad

        so_rev_outer = DI_STRAT.SecondOrder(AutoZygote(), AutoForwardDiff())
        @test DEER_STRAT._hvp_strategy(so_rev_outer) isa DEER_STRAT.ReverseOnGrad

        # mode-agnostic outer + forward-only inner → ReverseOverForward
        so_agnostic_outer = DI_STRAT.SecondOrder(AutoEnzyme(), AutoForwardDiff())
        @test DEER_STRAT._hvp_strategy(so_agnostic_outer) isa DEER_STRAT.ReverseOnGrad
    end

    @testset "the backend that runs is the one routed on" begin
        # both paths differentiate the already-built gradlogp, so both take the outer
        so_fwd = DI_STRAT.SecondOrder(AutoForwardDiff(), AutoZygote())
        @test DEER_STRAT._hvp_forward_backend(so_fwd) === AutoForwardDiff()

        so_rev = DI_STRAT.SecondOrder(AutoZygote(), AutoForwardDiff())
        @test DEER_STRAT._hvp_closure_backend(so_rev) === AutoZygote()
    end

    @testset "unwrapping a SecondOrder still reaches backend normalization" begin
        #= The outer half has to be taken before the backend-specific hook is
        dispatched on, or a wrapped `AutoEnzyme()` comes out bare: unnormalized,
        it lowers through reverse mode and aborts on GPU (see ext/EnzymeExt.jl). =#
        so = DI_STRAT.SecondOrder(AutoEnzyme(), AutoForwardDiff())
        @test DEER_STRAT._hvp_forward_backend(so) ===
            DEER_STRAT._hvp_forward_backend(AutoEnzyme())
        @test DEER_STRAT._hvp_closure_backend(so) ===
            DEER_STRAT._hvp_closure_backend(AutoEnzyme())
        @test DEER_STRAT._hvp_forward_backend(so).mode isa Enzyme.ForwardMode
        @test DEER_STRAT._hvp_closure_backend(so) isa AutoEnzyme{<:Any,Enzyme.Const}
    end

    @testset "strategy resolution is type-stable" begin
        @test @inferred(DEER_STRAT._hvp_strategy(AutoForwardDiff())) isa
            DEER_STRAT.ForwardOnGrad
        @test @inferred(DEER_STRAT._hvp_strategy(AutoMooncake())) isa
            DEER_STRAT.ReverseOnGrad
    end
end
