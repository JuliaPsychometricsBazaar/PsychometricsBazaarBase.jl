using PsychometricsBazaarBase.Integrators

@testset "scalar grid moments" begin
    for T in (Float32, Float64, BigFloat)
        x = T[1e8 - 1, 1e8 + 1]
        if T == Float32
            x = T[1e4 - 1, 1e4 + 1]
        end
        s = weighted_grid_moments(x, ones(T, 2))
        @test s.mass == 2
        @test s.mean == sum(x) / 2
        @test s.variance == 1
        @test s.mean isa T
        @test s.variance isa T
        x = T[-3, 1, 2]
        w = T[1, 2, 4]
        refmean = sum(BigFloat.(w) .* BigFloat.(x)) / sum(w)
        refvar = sum(BigFloat.(w) .* (BigFloat.(x) .- refmean).^2) / sum(w)
        s = weighted_grid_moments(x, w)
        @test s.mean ≈ T(refmean)
        @test s.variance ≈ T(refvar)
        scaled = weighted_grid_moments(x, 7w)
        @test scaled.mean ≈ s.mean
        @test scaled.variance ≈ s.variance
        @test weighted_grid_moments(x, w / sum(w)).variance ≈ s.variance
    end
    @test weighted_grid_moments([3.], [2.]) == (mass=2., mean=3., variance=0.)
    @test weighted_grid_moments([NaN, 2.], [0., 1.]).variance == 0
    f(x) = x == 99 ? error("outside support") : x^2
    @test weighted_grid_moments(f, [99, -2, 3], [0., 1., 0.]).mean == 4
    @test weighted_grid_moments([-1., 1.], [1., 1e-30]).variance ≈ 4e-30
    explicit = weighted_grid_moments([1., 3.], [1., 1.]; denominator=4.)
    @test explicit == (mass=2., mean=1., variance=1.)
    for shift in (0., -1e300)
        state = LogGridWeights(fill(shift, 2))
        s = weighted_grid_moments([1e8 - 1, 1e8 + 1], state)
        @test s.mean == 1e8
        @test s.variance == 1
        @test s.mass == grid_mass(state)
        @test s.log_scale == state.log_scale
        @test s.scaled_sum == state.scaled_sum
        @test weighted_grid_moments([1., 3.], state; denominator=2.).mean == 1
    end
    @test_throws ArgumentError weighted_grid_moments(Float64[], Float64[])
    @test_throws DimensionMismatch weighted_grid_moments([1.], [1., 2.])
    for w in ([0., 0.], [-1., 2.], [1., NaN], [1., Inf])
        @test_throws DomainError weighted_grid_moments([1., 2.], w)
    end
    for x in ([NaN, 1.], [Inf, 1.])
        @test_throws DomainError weighted_grid_moments(x, [1., 1.])
    end
    for d in (0., -1., Inf, NaN)
        @test_throws DomainError weighted_grid_moments([1.], [1.]; denominator=d)
    end
end

function moment_allocations(x, w)
    weighted_grid_moments(x, w)
    @allocated weighted_grid_moments(x, w)
end
@testset "warmed allocations" begin
    for n in (33, 65, 129)
        x = collect(range(-4., 4.; length=n))
        w = exp.(-x.^2 / 2)
        @test moment_allocations(x, w) == 0
    end
end
