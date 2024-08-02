using JET
using PsychometricsBazaarBase


@testset "JET checks" begin
    rep = report_package(
        PsychometricsBazaarBase;
        target_modules = (
            PsychometricsBazaarBase,
        ),
        mode = :typo
    )
    @show rep
    @test length(JET.get_reports(rep)) <= 0
    #@test_broken length(JET.get_reports(rep)) == 0
end
