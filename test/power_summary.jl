using Test
using PsychometricsBazaarBase
using Distributions

@testset "power_summary" begin
    @test power_summary(Normal()) == "Normal(μ=0.0, σ=1.0)"
end
