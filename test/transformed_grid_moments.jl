using PsychometricsBazaarBase.Integrators

@testset "transformed grid moment channels" begin
    for T in (Float32, Float64, BigFloat), n in (1, 2, 3), skip_checks in (false, true)
        grid = T[-3, 1, 2]
        weights = T[1, 2, 4]
        factors(x) = ntuple(k -> exp(x / T(k)), n)
        buffers = ntuple(_ -> similar(weights), n)
        for denominator in (nothing, T(4))
            actual = weighted_grid_moments!(factors, grid, weights, buffers; denominator, skip_checks)
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

@testset "unconditional producer calls and generic arrays" begin
    calls = Ref(0)
    factors(x) = begin
        calls[] += 1
        x == 1 ? (1.0, 0.0) : (0.0, 1.0)
    end
    grid = [99., 1., 3.]
    weights = [0., 2., 4.]
    scratch = (zeros(3), zeros(3))
    actual = weighted_grid_moments!(factors, grid, weights, scratch)
    @test calls[] == 3
    @test actual == ((mass=2., mean=1., variance=0.), (mass=4., mean=3., variance=0.))
    # Zero base and produced weights are valid on finite coordinates.
    @test weighted_grid_moments!(_ -> (1.,), [99., 2.], [0., 1.], (zeros(2),))[1].mean == 2
    @test weighted_grid_moments!(x -> (x == 99 ? 0. : 1.,), [99., 2.], [1., 1.], (zeros(2),))[1].mean == 2
    @test_throws ErrorException weighted_grid_moments!(
        x -> x == 99 ? error("called at zero weight") : (1.,),
        [99., 2.], [0., 1.], (zeros(2),))
    # Integer coordinates and abstract element types use the same reductions.
    generic = weighted_grid_moments!(x -> (x^2,), [-1, 1, 2], [1, 2, 3], (zeros(3),))
    @test generic[1] == weighted_grid_moments([-1, 1, 2], [1., 2., 12.])
    @test weighted_grid_moments!(_ -> (1.,), AbstractFloat[99., -1., 1.], [0., 1., 1.],
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
        @test_throws DomainError weighted_grid_moments!(pair, [1.], [1.], (zeros(1), zeros(1)); denominator=bad)
    end
    @test_throws DomainError weighted_grid_moments!(pair, [1.], [1.], (zeros(1), zeros(1)); denominator=0.)
    @test_throws DomainError weighted_grid_moments!(_ -> (0., 1.), [1.], [1.], (zeros(1), zeros(1)))
    @test_throws DomainError weighted_grid_moments!(_ -> (2.,), [1.], [floatmax()], (zeros(1),))
    input = [1.]
    @test_throws ArgumentError weighted_grid_moments!(pair, input, [1.], (input, zeros(1)))
    @test_throws ArgumentError weighted_grid_moments!(pair, [1.], input, (zeros(1), input))
    @test_throws ArgumentError weighted_grid_moments!(pair, [1.], [1.], (input, input))
end

@testset "centered reductions across vector lengths and layouts" begin
    for T in (Float32, Float64), n in (33, 65, 129, 257), strided in (false, true)
        offset = T == Float32 ? T(1e4) : T(1e12)
        grid = offset .+ T.(-(n ÷ 2):(n ÷ 2))
        weights = T[1 + mod(abs(i), 7) for i in -(n ÷ 2):(n ÷ 2)]
        # Zero weights occur at both ends and inside SIMD-sized chunks.
        for i in eachindex(grid)
            if mod(abs(i - (n ÷ 2 + 1)), 11) == 0 || i in (1, n)
                weights[i] = zero(T)
            end
        end
        if strided
            grid = view(repeat(grid; inner=2), 1:2:2n)
            weights = view(repeat(weights; inner=2), 1:2:2n)
        end
        buffers = ntuple(2) do _
            strided ? view(zeros(T, 2n), 1:2:2n) : zeros(T, n)
        end
        seen = T[]
        factors(x) = begin
            push!(seen, x)
            (one(T), isodd(round(Int, x - offset)) ? T(2) : zero(T))
        end
        for denominator in (nothing, T(4))
            empty!(seen)
            summaries = weighted_grid_moments!(factors, grid, weights, buffers; denominator)
            @test seen == grid
            for (buffer, actual) in zip(buffers, summaries)
                expected = weighted_grid_moments(BigFloat.(grid), BigFloat.(buffer);
                    denominator=isnothing(denominator) ? nothing : BigFloat(denominator))
                @test actual.mass ≈ expected.mass rtol=8eps(T)
                @test actual.mean ≈ expected.mean rtol=8eps(T)
                @test actual.variance ≈ expected.variance rtol=8eps(T)
            end
        end
    end
end

@testset "all weights preserve callback order" begin
    for T in (Float32, Float64, BigFloat), channels in (1, 2, 3),
            strided in (false, true), denominator in (nothing, T(4))
        xs = collect(range(T(-4), T(4); length=33))
        grid = strided ? view(repeat(xs; inner=2), 1:2:66) : copy(xs)
        weights = exp.(-grid.^2)
        buffers = ntuple(channels) do _
            strided ? view(zeros(T, 66), 1:2:66) : zeros(T, 33)
        end
        # Reuse buffers while switching from dense to sparse support and back.
        for sparse in (false, true, false)
            grid[1] = xs[1]
            weights[1] = sparse ? zero(T) : exp(-xs[1]^2)
            seen = T[]
            factors = x -> (push!(seen, x); ntuple(k -> exp(x / T(k)), channels))
            actual = weighted_grid_moments!(factors, grid, weights, buffers; denominator)
            @test seen == grid
            for (buffer, result) in zip(buffers, actual)
                expected = weighted_grid_moments(grid, buffer; denominator)
                for field in (:mass, :mean, :variance)
                    @test isapprox(getproperty(result, field), getproperty(expected, field);
                        rtol=64eps(T), atol=64eps(T))
                end
            end
        end
    end
end

function transformed_allocations(grid, weights, buffers, skip_checks)
    producer(x) = (exp(x), exp(-x))
    weighted_grid_moments!(producer, grid, weights, buffers; skip_checks)
    @allocated weighted_grid_moments!(producer, grid, weights, buffers; skip_checks)
end
@testset "scalar moments reuse scratch" begin
    for T in (Float32, Float64), n in (33, 65, 129), skip_checks in (false, true)
        grid = collect(range(T(-4), T(4); length=n))
        weights = exp.(-grid.^2 / 2)
        @test transformed_allocations(grid, weights, (similar(weights), similar(weights)), skip_checks) == 0
    end
end

# Observe shape checks on valid storage, without exercising unsafe indexing.
struct MomentAxesProbe{T} <: AbstractVector{T}
    data::Vector{T}
    calls::Base.RefValue{Int}
    alias_calls::Base.RefValue{Int}
end
MomentAxesProbe(data, calls) = MomentAxesProbe(data, calls, Ref(0))
Base.size(v::MomentAxesProbe) = size(v.data)
Base.length(v::MomentAxesProbe) = length(v.data)
Base.axes(v::MomentAxesProbe) = (v.calls[] += 1; axes(v.data))
Base.eachindex(v::MomentAxesProbe) = eachindex(v.data)
Base.getindex(v::MomentAxesProbe, i::Int) = v.data[i]
Base.dataids(v::MomentAxesProbe) = (v.alias_calls[] += 1; Base.dataids(v.data))

struct MomentFactorProbe
    calls::Base.RefValue{Int}
end
Base.length(v::MomentFactorProbe) = (v.calls[] += 1; 2)
Base.getindex(::MomentFactorProbe, i::Int) = (1., 2.)[i]

function bounds_moments(factors::F, grid, weights, buffers, ::Val{Inbounds};
        denominator=nothing) where {F, Inbounds}
    if Inbounds
        @inbounds weighted_grid_moments!(factors, grid, weights, buffers; denominator)
    else
        weighted_grid_moments!(factors, grid, weights, buffers; denominator)
    end
end

@testset "shape elision retains numerical and alias checks" begin
    for T in (Float64, Int, AbstractFloat), inbounds in (false, true)
        axes_calls, factor_calls = Ref(0), Ref(0)
        grid = MomentAxesProbe(T[-1, 0, 1], axes_calls)
        weights = ones(3)
        buffers = (zeros(3), zeros(3))
        factors = _ -> MomentFactorProbe(factor_calls)
        result = bounds_moments(factors, grid, weights, buffers, Val(inbounds))
        @test result == ((mass=3., mean=0., variance=2/3), (mass=6., mean=0., variance=2/3))
        # --check-bounds=yes must still force the checks, including under @inbounds.
        checked = !inbounds || Base.JLOptions().check_bounds == 1
        @test (axes_calls[] > 0) == checked
        @test factor_calls[] == (checked ? 3 : 0)
        @test_throws ArgumentError bounds_moments(factors, grid, weights,
            (weights, zeros(3)), Val(inbounds))
        @test_throws ArgumentError bounds_moments(factors, grid, weights,
            (buffers[1], buffers[1]), Val(inbounds))
        @test_throws DomainError bounds_moments(_ -> (0., 1.), grid, weights,
            buffers, Val(inbounds))
        @test_throws DomainError bounds_moments(_ -> (floatmax(), floatmax()),
            grid, weights, buffers, Val(inbounds))
        for denominator in (0., -1., Inf, NaN)
            @test_throws DomainError bounds_moments(factors, grid, weights,
                buffers, Val(inbounds); denominator)
        end
    end
end

@testset "skip_checks bypasses contracts but retains mass and denominator checks" begin
    for T in (Float64, Int, AbstractFloat), skip_checks in (false, true)
        axes_calls, factor_calls = Ref(0), Ref(0)
        grid = MomentAxesProbe(T[-1, 0, 1], axes_calls)
        weights = [0., 1., 2.]
        buffers = (zeros(3), zeros(3))
        factors = _ -> MomentFactorProbe(factor_calls)
        result = weighted_grid_moments!(factors, grid, weights, buffers; skip_checks)
        @test result[1].mean ≈ 2/3
        @test result[1].variance ≈ 2/9
        @test buffers == ([0., 1., 2.], [0., 2., 4.])
        @test (axes_calls[] > 0) == !skip_checks
        @test (grid.alias_calls[] > 0) == !skip_checks
        @test factor_calls[] == (skip_checks ? 0 : 3)
        for base in (zeros(3), weights)
            @test_throws DomainError weighted_grid_moments!(
                _ -> (0., 1.), grid, base, buffers; skip_checks)
        end
        @test_throws DomainError weighted_grid_moments!(
            _ -> (floatmax(), floatmax()), grid, weights, buffers; skip_checks)
        for denominator in (0., -1., Inf, NaN)
            @test_throws DomainError weighted_grid_moments!(
                factors, grid, weights, buffers; skip_checks, denominator)
        end
    end
end
