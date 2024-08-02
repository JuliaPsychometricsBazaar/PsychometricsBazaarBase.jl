using JuliaFormatter
using PsychometricsBazaarBase

@testcase "format" begin
    dir = pkgdir(PsychometricsBazaarBase)
    @test format(dir * "/src"; overwrite = false)
    @test format(dir * "/test"; overwrite = false)
end
