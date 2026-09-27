using PsychometricsBazaarBase
using PsychometricsBazaarBase: IntegralCoeffs, Integrators, Optimizers
using PsychometricsBazaarBase.Integrators
using PsychometricsBazaarBase.Optimizers: OneDimOptimOptimizer, MultiDimOptimOptimizer,
                                        FixedGridOptimizer
using Optim: NelderMead

for T in (Float32, BigFloat)
    lo, hi = zero(T), one(T)
    @test IntegralCoeffs.one(lo) == hi
    @test IntegralCoeffs.one(lo) isa T
    @test IntegralCoeffs.one(T[lo]) == hi
    @test IntegralCoeffs.one(T[lo]) isa T

    grid = Integrators.even_grid(lo, hi, 3)
    @test eltype(grid.grid) === T
    @test intval(grid(identity)) == T(1.5)
    @test intval(grid(identity)) isa T
    @test eltype(preallocate(grid).buf) === T
    @test intval(MidpointIntegrator(grid.grid)(identity)) isa T
    @test intval(IterativeFixedGridIntegrator(grid.grid)(identity)) isa T

    multidim_grid = Integrators.even_grid(T[lo, lo], T[hi, hi], 3)
    @test eltype(preallocate(multidim_grid).buf) === T
    @test intval(multidim_grid(x -> x[1] + x[2])) isa T

    @test intval(FixedGKIntegrator(lo, hi, 7)(identity)) ≈ T(0.5)
    @test intval(FixedGKIntegrator(lo, hi, 7)(identity)) isa T
    multidim_gk = MultiDimFixedGKIntegrator(T[lo, lo], T[hi, hi], 3)
    @test intval(multidim_gk(x -> x[1] * x[2])) ≈ T(0.25)
    @test intval(multidim_gk(x -> x[1] * x[2])) isa T

    @test intval(HCubatureIntegrator(T[lo, lo], T[hi, hi])(
        x -> x[1] * x[2])) isa T
    cubature = CubatureIntegrator(T[lo, lo], T[hi, hi])
    @test cubature.lo isa Vector{T}
    @test intval(cubature(x -> x[1] * x[2])) ≈ 0.25
    cuba = CubaIntegrator(T[lo, lo], T[hi, hi], CubaCuhre())
    @test cuba.lo isa Vector{T}
    @test intval(cuba(x -> x[1] * x[2])) ≈ 0.25

    fixed_optimizer = Optimizers.even_grid(lo, hi, 3)
    @test fixed_optimizer isa FixedGridOptimizer
    @test fixed_optimizer(x -> x) == lo
    one_dim_optimizer = OneDimOptimOptimizer(-hi, hi, NelderMead())
    @test one_dim_optimizer.lo isa T
    @test preallocate(one_dim_optimizer)(x -> -x^2) isa T
    multi_dim_optimizer = MultiDimOptimOptimizer(T[-hi, -hi], T[hi, hi], NelderMead())
    @test eltype(preallocate(multi_dim_optimizer).x) === T
end

@test intval(FixedGridIntegrator([0, 1, 2])(x -> x / 2)) == 1.5
@test intval(MidpointIntegrator([0, 1, 2])(x -> x / 2)) == 2.0
