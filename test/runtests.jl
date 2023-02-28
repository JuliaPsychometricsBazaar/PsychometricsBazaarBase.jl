using Test

@testset "Aqua automated quality checks" begin
    include("./aqua.jl")
end

@testset "Smoke tests" begin
    include("./smoke.jl")
end
