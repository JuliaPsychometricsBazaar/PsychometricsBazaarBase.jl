using XUnit

@testset "aqua" begin
    include("./aqua.jl")
end

@testset "jet" begin
    include("./jet.jl")
end

@testset "smoke" begin
    include("./smoke.jl")
end

@testset "format" begin
    include("./format.jl")
end
