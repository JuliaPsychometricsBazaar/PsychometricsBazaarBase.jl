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

@testset "indentwrappers" begin
    include("./indentwrappers.jl")
end

@testset "power summary" begin
    include("./power_summary.jl")
end
