using Test
using Random
using ParallelMCMC
const DEER = ParallelMCMC.DEER

@testset "affine scan diag matches sequential recursion" begin
    rng = MersenneTwister(1)
    D = 5
    T = 64
    s0 = randn(rng, D)

    a = [0.9 .+ 0.02 .* randn(rng, D) for _ in 1:T] # stable-ish
    b = [0.1 .* randn(rng, D) for _ in 1:T]

    # Sequential reference
    s_ref = Vector{Vector{Float64}}(undef, T)
    s_prev = copy(s0)
    for t in 1:T
        s_prev = a[t] .* s_prev .+ b[t]
        s_ref[t] = copy(s_prev)
    end

    # Scan
    s_scan = DEER.solve_affine_scan_diag(a, b, s0)

    @test length(s_scan) == T
    for t in 1:T
        @test isapprox(s_scan[t], s_ref[t]; rtol=1e-12, atol=1e-12)
    end
end
