using JET
using PsychometricsBazaarBase

@testset "JET checks" begin
    rep = report_package(
        PsychometricsBazaarBase;
        target_modules = (
            PsychometricsBazaarBase,
        ),
        ignored_modules = (
            PsychometricsBazaarBase.Parameters,
        ),
        mode = :typo
    )
    @show rep
    @test length(JET.get_reports(rep)) <= 0
    #@test_broken length(JET.get_reports(rep)) == 0
end
