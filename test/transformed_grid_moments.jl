using PsychometricsBazaarBase.Integrators

@testset "transformed grid moment channels" begin
    for T in (Float32, Float64, BigFloat), n in (1, 2, 3)
        grid = T[-3, 1, 2]
        weights = T[1, 2, 4]
        factors(x) = ntuple(k -> exp(x / T(k)), n)
        buffers = ntuple(_ -> similar(weights), n)
        for denominator in (nothing, T(4))
            actual = weighted_grid_moments!(factors, grid, weights, buffers; denominator)
            for k in 1:n
                expected_weights = weights .* [factors(x)[k] for x in grid]
                expected = weighted_grid_moments(grid, expected_weights; denominator)
                @test buffers[k] == expected_weights
                @test actual[k].mass ≈ expected.mass
                @test actual[k].mean ≈ expected.mean
                @test actual[k].variance ≈ expected.variance
                @test actual[k].variance isa T
            end
        end
        offset = T == Float32 ? T(1e4) : T(1e8)
        xs = [offset - 1, offset + 1]
        actual = weighted_grid_moments!(_ -> (one(T),), xs, ones(T, 2), (zeros(T, 2),))
        @test actual[1].mean == offset
        @test actual[1].variance == 1
    end
end

@testset "support, producer calls and generic arrays" begin
    calls = Ref(0)
    factors(x) = begin
        calls[] += 1
        x == 99 && error("outside support")
        x == 1 ? (1.0, 0.0) : (0.0, 1.0)
    end
    grid = [99., 1., 3.]
    weights = [0., 2., 4.]
    scratch = (zeros(3), zeros(3))
    actual = weighted_grid_moments!(factors, grid, weights, scratch)
    @test calls[] == 2
    @test actual == ((mass=2., mean=1., variance=0.), (mass=4., mean=3., variance=0.))
    # Zero-weight NaN coordinates do not enter centered arithmetic.
    @test weighted_grid_moments!(_ -> (1.,), [NaN, 2.], [0., 1.], (zeros(2),))[1].mean == 2
    @test weighted_grid_moments!(x -> (isnan(x) ? 0. : 1.,), [NaN, 2.], [1., 1.], (zeros(2),))[1].mean == 2
    # Integer coordinates exercise the generic materialization fallback.
    generic = weighted_grid_moments!(x -> (x^2,), [-1, 1, 2], [1, 2, 3], (zeros(3),))
    @test generic[1] == weighted_grid_moments([-1, 1, 2], [1., 2., 12.])
    @test weighted_grid_moments!(_ -> (1.,), AbstractFloat[NaN, -1., 1.], [0., 1., 1.],
        (AbstractFloat[0., 0., 0.],))[1].variance == 1
    # Conservatively overlapping views are intentionally rejected by mightalias;
    # use independent storage for arbitrary strided arrays.
    views = (view(zeros(6), 1:2:5), view(zeros(6), 2:2:6))
    @test weighted_grid_moments!(_ -> (1., 1.), [-1., 0., 1.], ones(3), views)[1].variance ≈ 2 / 3
end

@testset "transformed moment input validation" begin
    pair(x) = (1., 1.)
    @test_throws ArgumentError weighted_grid_moments!(pair, Float64[], Float64[], (Float64[], Float64[]))
    @test_throws ArgumentError weighted_grid_moments!(pair, [1.], [1.], ())
    @test_throws DimensionMismatch weighted_grid_moments!(pair, [1.], [1., 2.], (zeros(1), zeros(1)))
    @test_throws DimensionMismatch weighted_grid_moments!(pair, [1.], [1.], (zeros(2), zeros(1)))
    @test_throws DimensionMismatch weighted_grid_moments!(_ -> (1.,), [1.], [1.], (zeros(1), zeros(1)))
    for bad in (-1., NaN, Inf)
        @test_throws DomainError weighted_grid_moments!(pair, [1.], [bad], (zeros(1), zeros(1)))
        @test_throws DomainError weighted_grid_moments!(_ -> (1., bad), [1.], [1.], (zeros(1), zeros(1)))
        @test_throws DomainError weighted_grid_moments!(pair, [1.], [1.], (zeros(1), zeros(1)); denominator=bad)
    end
    @test_throws DomainError weighted_grid_moments!(pair, [1.], [1.], (zeros(1), zeros(1)); denominator=0.)
    @test_throws DomainError weighted_grid_moments!(_ -> (0., 1.), [1.], [1.], (zeros(1), zeros(1)))
    @test_throws DomainError weighted_grid_moments!(pair, [NaN], [1.], (zeros(1), zeros(1)))
    @test_throws DomainError weighted_grid_moments!(_ -> (2.,), [1.], [floatmax()], (zeros(1),))
    input = [1.]
    @test_throws ArgumentError weighted_grid_moments!(pair, input, [1.], (input, zeros(1)))
    @test_throws ArgumentError weighted_grid_moments!(pair, [1.], input, (zeros(1), input))
    @test_throws ArgumentError weighted_grid_moments!(pair, [1.], [1.], (input, input))
end

function transformed_allocations(grid, weights, buffers)
    producer(x) = (exp(x), exp(-x))
    weighted_grid_moments!(producer, grid, weights, buffers)
    @allocated weighted_grid_moments!(producer, grid, weights, buffers)
end
@testset "fused scalar moments reuse scratch" begin
    for T in (Float32, Float64), n in (33, 65, 129)
        grid = collect(range(T(-4), T(4); length=n))
        weights = exp.(-grid.^2 / 2)
        @test transformed_allocations(grid, weights, (similar(weights), similar(weights))) == 0
    end
end
