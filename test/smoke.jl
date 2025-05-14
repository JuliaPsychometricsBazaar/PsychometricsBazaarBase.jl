using PsychometricsBazaarBase.Integrators
using PsychometricsBazaarBase.Optimizers
using Optim

const things = [
    OneDimOptimOptimizer(-6.0, 6.0, NelderMead()),
    QuadGKIntegrator(; lo=-6.0, hi=6.0, order=5),
    FixedGKIntegrator(-6.0, 6.0, 80)
]

for thing in things
    @test thing(x -> x) isa AbstractIntegrationResult
end
