using JuliaFormatter
using PsychometricsBazaarBase

@testset "format" begin
    dir = pkgdir(PsychometricsBazaarBase)
    @test format(dir * "/src"; overwrite = false)
    @test format(dir * "/test"; overwrite = false)
end
