using Test

@testset "aqua" begin
    include("./aqua.jl")
end

@testset "jet" begin
    include("./jet.jl")
end

@testset "smoke" begin
    include("./smoke.jl")
end

@testset "optimizers" begin
    include("./optimizers.jl")
end

@testset "indentwrappers" begin
    include("./indentwrappers.jl")
end

@testset "power summary" begin
    include("./power_summary.jl")
end

@testset "log grid weights" begin
    include("./log_grid_weights.jl")
end

@testset "grid moments" begin
    include("./grid_moments.jl")
end
