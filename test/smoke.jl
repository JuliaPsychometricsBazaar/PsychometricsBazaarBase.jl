using PsychometricsBazzarBase.Integrators
using PsychometricsBazzarBase.Optimizers
using Optim


const things = [
    OneDimOptimOptimizer(-6.0, 6.0, NelderMead()),
    QuadGKIntegrator(-6, 6, 5),
    FixedGKIntegrator(-6, 6, 80),
]

for thing in things
    thing(x -> x)
end
