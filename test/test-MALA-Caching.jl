using Test
using Random
using LinearAlgebra

using ParallelMCMC
const MALA_c = ParallelMCMC.MALA
const AbstractMCMC_c = ParallelMCMC.AbstractMCMC

# Not Gaussian, so the gradient depends on x and some proposals are rejected.
logp_c(x) = -0.5 * dot(x, x) - 0.1 * sum(abs2.(x) .^ 2)
grad_c(x) = @. -x - 0.4 * x^3

function counting_model(D)
    n_logp, n_grad = Ref(0), Ref(0)
    lp(x) = (n_logp[] += 1; logp_c(x))
    gr(x) = (n_grad[] += 1; grad_c(x))
    return DensityModel(lp, gr, D), n_logp, n_grad
end

function run_chain(model, sampler, N, x0)
    return sample(
        MersenneTwister(1),
        model,
        sampler,
        N;
        chain_type=Any,
        progress=false,
        initial_params=x0,
    )
end

@testset "MALASampler matches the uncached kernel" begin
    x0 = [0.3, -1.2, 2.0]
    ε, N = 0.2, 300
    chain = run_chain(DensityModel(logp_c, grad_c, 3), MALASampler(ε), N, x0)

    # Same random numbers, but `logp` and `gradlogp` are evaluated at `x` again each step.
    rng = MersenneTwister(1)
    ws = MALA_c.MALAWorkspace(x0)
    x = copy(x0)
    @test chain[1].x == x0
    for i in 2:N
        ξ = randn(rng, 3)
        u = rand(rng)
        x, accepted, _ = MALA_c.mala_step_with_logα!(
            similar(x), ws, logp_c, grad_c, x, ε, ξ, u
        )
        @test chain[i].x == x
        @test chain[i].accepted == accepted
        @test chain[i].logp == logp_c(x)
    end
end

@testset "$(nameof(typeof(sampler))) evaluates logp and grad once per step" for sampler in (
    MALASampler(0.2), AdaptiveMALASampler(0.2; n_warmup=50)
)
    N = 200
    model, n_logp, n_grad = counting_model(3)
    run_chain(model, sampler, N, [0.3, -1.2, 2.0])
    # One evaluation for the initial point and one per proposal. Before caching it was
    # about 3N for `logp` and 2N for `grad`.
    @test n_logp[] <= N + 1
    @test n_grad[] <= N + 1
end

@testset "a state from another model gets that model's log-density" begin
    x0 = [0.3, -1.2, 2.0]
    model_a = DensityModel(logp_c, grad_c, 3)
    model_b = DensityModel(x -> logp_c(x) + 5, grad_c, 3)
    for sampler in (MALASampler(0.2), AdaptiveMALASampler(0.2; n_warmup=10))
        _, state = AbstractMCMC_c.step(
            MersenneTwister(2), model_a, sampler; initial_params=x0
        )
        for seed in 1:20
            t, _ = AbstractMCMC_c.step(MersenneTwister(seed), model_b, sampler, state)
            @test t.logp ≈ logp_c(t.x) + 5
        end
    end
end
