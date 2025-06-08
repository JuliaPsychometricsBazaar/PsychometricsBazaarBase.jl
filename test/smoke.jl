using PsychometricsBazaarBase.Integrators
using PsychometricsBazaarBase.Optimizers
using Optim

const integrators = [
    QuadGKIntegrator(; lo=-6.0, hi=6.0, order=5),
    FixedGKIntegrator(-6.0, 6.0, 80)
]

for integrator in integrators
    @test integrator(x -> x) isa AbstractIntegrationResult
end

@test OneDimOptimOptimizer(-6.0, 6.0, NelderMead())(x -> x) isa Number