using PsychometricsBazaarBase: Optimizers
using PsychometricsBazaarBase.Optimizers
using PsychometricsBazaarBase.Optimizers: NativeOneDimOptimOptimizer
using Optim

@testset "one dimensional optimizer respects its bounds" begin
    # The maximum is outside the box
    f = x -> -(x - 8.0)^2
    for optim in (NelderMead(), BFGS(), IPNewton())
        @test OneDimOptimOptimizer(-6.0, 6.0, optim)(f) <= 6.0
    end
    # Note that Optim's NelderMead is degenerate in one dimension, so it stalls
    # well inside the box rather than converging on the boundary
    @test OneDimOptimOptimizer(-6.0, 6.0, BFGS())(f)≈6.0 atol=1e-2
end

@testset "multi dimensional optimizer respects its bounds" begin
    f = x -> -sum((x .- 8.0) .^ 2)
    for optim in (NelderMead(), BFGS())
        res = MultiDimOptimOptimizer([-6.0, -6.0], [6.0, 6.0], optim)(f)
        @test all(res .<= 6.0)
        @test res≈[6.0, 6.0] atol=1e-2
    end
end

@testset "preallocated one dimensional optimizer" begin
    opt = OneDimOptimOptimizer(-6.0, 6.0, NelderMead())
    prealloc = preallocate(opt)
    @test prealloc isa PreallocatedOptimOptimizer
    for c in (0.3, -1.5, 2.0, 8.0)
        f = x -> -(x - c)^2
        @test prealloc(f) ≈ opt(f)
    end
    # Reruns of the same function must not reuse the previous run's result
    f = x -> -(x - 0.4)^2
    @test prealloc(f) == prealloc(f)
end

@testset "preallocated multi dimensional optimizer" begin
    opt = MultiDimOptimOptimizer([-6.0, -6.0], [6.0, 6.0], NelderMead())
    prealloc = preallocate(opt)
    @test prealloc isa PreallocatedOptimOptimizer
    for c in ([0.3, 0.1], [-1.5, 2.0], [8.0, 8.0])
        f = x -> -sum((x .- c) .^ 2)
        @test prealloc(f) ≈ opt(f)
    end
end

@testset "preallocated first order optimizer" begin
    opt = MultiDimOptimOptimizer([-6.0, -6.0], [6.0, 6.0], BFGS())
    prealloc = preallocate(opt)
    @test prealloc isa PreallocatedOptimOptimizer
    for c in ([0.3, 0.1], [-1.5, 2.0])
        f = x -> -sum((x .- c) .^ 2)
        @test prealloc(f)≈opt(f) atol=1e-6
        @test prealloc(f)≈c atol=1e-4
    end
end

@testset "preallocated optimizer within an infinite box" begin
    opt = MultiDimOptimOptimizer([-Inf, -Inf], [Inf, Inf], [0.0, 0.0], NelderMead(),
        Optim.Options())
    prealloc = preallocate(opt)
    @test prealloc isa PreallocatedOptimOptimizer
    for c in ([0.3, 0.1], [-1.5, 2.0])
        f = x -> -sum((x .- c) .^ 2)
        @test prealloc(f) == opt(f)
    end
    # A box passed per call falls back to the non-preallocated path
    f = x -> -sum((x .- 8.0) .^ 2)
    @test all(prealloc(f; lo = [-6.0, -6.0], hi = [6.0, 6.0]) .<= 6.0)
end

@testset "optimizers with nothing to preallocate" begin
    @test preallocate(OneDimOptimOptimizer(-6.0, 6.0, IPNewton())) isa OneDimOptimOptimizer
    @test preallocate(NativeOneDimOptimOptimizer(; lo = -6.0, hi = 6.0)) isa
          NativeOneDimOptimOptimizer
    @test preallocate(Optimizers.even_grid(-6.0, 6.0, 21)) isa FixedGridOptimizer
end
