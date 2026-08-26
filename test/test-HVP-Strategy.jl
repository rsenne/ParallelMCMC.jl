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

    @testset "AutoReactant short-circuits hvp_mode" begin
        #= DI cannot drive Reactant, so `AutoReactant` never reaches `DI.hvp_mode`
        and the strategy is picked by dispatch instead. The method lives in DEER
        rather than in ReactantExt, so this holds with Reactant unloaded. =#
        @test DEER_STRAT._hvp_strategy(AutoReactant()) isa DEER_STRAT.ReactantHVP
        @test @inferred(DEER_STRAT._hvp_strategy(AutoReactant())) isa DEER_STRAT.ReactantHVP
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

    @testset "normalization supplies Const but never a mode" begin
        #= The wrappers DEER differentiates are its own types, so annotating them
        `Const` is its business. The mode is not. =#
        bare = DEER_STRAT._normalized_backend(AutoEnzyme())
        @test bare isa AutoEnzyme{<:Any,Enzyme.Const}
        @test bare.mode === nothing

        for mode in (Enzyme.Forward, Enzyme.Reverse)
            normalized = DEER_STRAT._normalized_backend(AutoEnzyme(; mode=mode))
            @test normalized.mode === mode
            @test normalized isa AutoEnzyme{<:Any,Enzyme.Const}
        end

        # An annotation the user chose is left alone.
        annotated = AutoEnzyme(; function_annotation=Enzyme.Duplicated)
        @test DEER_STRAT._normalized_backend(annotated) === annotated

        # Backends with no specialization pass straight through.
        @test DEER_STRAT._normalized_backend(AutoForwardDiff()) === AutoForwardDiff()
        @test DEER_STRAT._normalized_backend(AutoZygote()) === AutoZygote()
    end

    @testset "normalizing a SecondOrder keeps the composition DI resolved" begin
        #= Regression for #62. Normalization used to route the outer half through
        a forward-only hook, which pinned `Enzyme.Forward` onto it. For a pair
        `hvp_mode` resolves to reverse — `SecondOrder(AutoEnzyme(),
        AutoForwardDiff())` is reverse-over-forward, its inner half being
        forward-only — that made it forward-over-forward. =#
        for so in (
            DI_STRAT.SecondOrder(AutoEnzyme(), AutoForwardDiff()),
            DI_STRAT.SecondOrder(AutoEnzyme(), AutoZygote()),
            DI_STRAT.SecondOrder(AutoEnzyme(; mode=Enzyme.Reverse), AutoForwardDiff()),
            DI_STRAT.SecondOrder(AutoEnzyme(; mode=Enzyme.Forward), AutoZygote()),
            DI_STRAT.SecondOrder(AutoForwardDiff(), AutoZygote()),
            DI_STRAT.SecondOrder(AutoZygote(), AutoForwardDiff()),
        )
            normalized = DEER_STRAT._normalized_second_order(so)
            @test DI_STRAT.hvp_mode(normalized) == DI_STRAT.hvp_mode(so)
            # The inner half is the user's own first-order gradient, untouched.
            @test DI_STRAT.inner(normalized) === DI_STRAT.inner(so)
            # And no mode is invented for the outer half either. Only `AutoEnzyme`
            # carries a `mode`, so this is the case that could regress.
            if DI_STRAT.outer(so) isa AutoEnzyme
                @test DI_STRAT.outer(normalized).mode === DI_STRAT.outer(so).mode
                #= The outer half still reaches EnzymeExt's specialization rather
                than passing through as a bare backend. =#
                @test DI_STRAT.outer(normalized) isa AutoEnzyme{<:Any,Enzyme.Const}
            end
        end
    end

    @testset "strategy resolution is type-stable" begin
        @test @inferred(DEER_STRAT._hvp_strategy(AutoForwardDiff())) isa
            DEER_STRAT.ForwardOnGrad
        @test @inferred(DEER_STRAT._hvp_strategy(AutoMooncake())) isa
            DEER_STRAT.ReverseOnGrad
    end
end
