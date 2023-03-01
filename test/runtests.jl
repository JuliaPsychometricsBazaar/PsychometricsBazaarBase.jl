using XUnit


@testset runner=ParallelTestRunner() xml_report=true "top" begin
    @testset "aqua" begin
        include("./aqua.jl")
    end

    @testset "smoke" begin
	include("./smoke.jl")
    end
end
