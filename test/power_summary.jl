using Test
using PsychometricsBazaarBase
using Distributions

@testset "power_summary" begin
    @test power_summary(Normal()) in ("Normal(μ=0.0, σ=1.0)", "Normal(0.0, 1.0)")
end
