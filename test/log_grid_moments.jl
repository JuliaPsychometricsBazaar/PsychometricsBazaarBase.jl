using Test
using PsychometricsBazaarBase: Integrators

@testset "transformed log moments" begin
    for T in (Float32, Float64, BigFloat), n in (1, 33, 65), channels in (1, 2, 3, 5)
        grid = n == 1 ? T[0] : collect(range(T(-4), T(4); length=n))
        logs = -grid.^2 .- T(1000)
        log_buffers = ntuple(_ -> similar(logs), channels)
        buffers = ntuple(_ -> similar(logs), channels)
        factors(x) = ntuple(k -> -T(k) * (x - T(0.5))^2, channels)
        calls = T[]
        counted(x) = (push!(calls, x); factors(x))
        actual = Integrators.weighted_log_grid_moments!(counted, grid, logs, log_buffers, buffers)
        @test calls == grid
        for k in 1:channels
            reference = Integrators.weighted_grid_moments(grid,
                Integrators.LogGridWeights(logs + [factors(x)[k] for x in grid]))
            @test actual[k].mean ≈ reference.mean rtol=128eps(T) atol=128eps(T)
            @test actual[k].variance ≈ reference.variance rtol=128eps(T) atol=128eps(T)
            @test actual[k].log_scale == reference.log_scale
            @test actual[k].scaled_sum ≈ reference.scaled_sum rtol=128eps(T)
            @test actual[k].mean isa T
            @test buffers[k] ≈ exp.(log_buffers[k] .- actual[k].log_scale)
        end
        unchecked = Integrators.weighted_log_grid_moments!(factors, grid, logs,
            log_buffers, buffers; skip_checks=true)
        for k in 1:channels
            @test collect(values(unchecked[k])) ≈ collect(values(actual[k])) rtol=128eps(T)
        end
    end
end

@testset "log mass, support and centered variance" begin
    grid = [1e8 - 1, 1e8 + 1]
    logs = [-1e200, -1e200]
    scratch = (similar(logs), similar(logs))
    weights = (similar(logs), similar(logs))
    result = Integrators.weighted_log_grid_moments!(x -> (0., 0.), grid, logs, scratch, weights)
    @test all(s -> s.mean == 1e8 && s.variance == 1., result)
    # A tail lost by normalizing history first can dominate after conditioning.
    grid = [0., 1.]
    logs = [0., -1000.]
    result = Integrators.weighted_log_grid_moments!(x -> (x == 0 ? -2000. : 0.,),
        grid, logs, (zeros(2),), (zeros(2),))
    @test only(result).mean == 1.
    @test only(result).variance == 0.
    result = Integrators.weighted_log_grid_moments!(x -> (0.,), grid, [0., -Inf],
        (zeros(2),), (zeros(2),))
    @test only(result).mean == 0.
    for skip_checks in (false, true)
        @test_throws DomainError Integrators.weighted_log_grid_moments!(x -> (-Inf,), grid,
            zeros(2), (zeros(2),), (zeros(2),); skip_checks)
    end
end

@testset "log moment buffer contracts" begin
    grid = [-1., 0., 1.]; logs = zeros(3)
    f(x) = (0., 0.)
    @test_throws DimensionMismatch Integrators.weighted_log_grid_moments!(f, grid, logs,
        (zeros(2), zeros(3)), (zeros(3), zeros(3)))
    @test_throws DimensionMismatch Integrators.weighted_log_grid_moments!(x -> (0.,), grid, logs,
        (zeros(3), zeros(3)), (zeros(3), zeros(3)))
    shared = zeros(3)
    @test_throws ArgumentError Integrators.weighted_log_grid_moments!(f, grid, logs,
        (shared, zeros(3)), (zeros(3), shared))
    @test_throws ArgumentError Integrators.weighted_log_grid_moments!(f, grid, logs,
        (logs, zeros(3)), (zeros(3), zeros(3)))
    @test_throws ArgumentError Integrators.weighted_log_grid_moments!(x -> (), grid, logs, (), ())
    @test_throws ArgumentError Integrators.weighted_log_grid_moments!(x -> (0.,), Float64[], Float64[],
        (Float64[],), (Float64[],))
end

function log_moment_allocations()
    grid = collect(range(-4.,4.;length=33)); logs = -grid.^2
    scratch = (similar(logs),similar(logs)); weights = (similar(logs),similar(logs))
    factors(x) = (-x^2, -2x^2)
    Integrators.weighted_log_grid_moments!(factors,grid,logs,scratch,weights;skip_checks=true)
    @allocated Integrators.weighted_log_grid_moments!(factors,grid,logs,scratch,weights;skip_checks=true)
end
@testset "log moment workspace reuse" begin
    @test log_moment_allocations() == 0
    # Independent strided storage exercises the generic array loop.
    grid = [-1.,0.,1.]; logs = zeros(3)
    scratch = (view(zeros(6),1:2:5),); weights = (view(zeros(6),2:2:6),)
    @test only(Integrators.weighted_log_grid_moments!(x -> (0.,),grid,logs,scratch,weights)).variance ≈ 2/3
end

@testset "log buffers define stored precision" begin
    grid = [-1.,0.,1.]; logs = [-1000.1,-1000.2,-1000.3]
    scratch = (zeros(Float32,3), zeros(3)); weights = (zeros(Float32,3),zeros(3))
    summaries = Integrators.weighted_log_grid_moments!(x -> (x/3, x/3),grid,logs,scratch,weights)
    for k in 1:2
        reference = Integrators.weighted_grid_moments(grid, Integrators.LogGridWeights(scratch[k]))
        @test summaries[k].log_scale == maximum(scratch[k])
        @test summaries[k].scaled_sum ≈ sum(weights[k])
        @test summaries[k].mean ≈ reference.mean rtol=32eps(eltype(scratch[k]))
        @test summaries[k].variance ≈ reference.variance rtol=32eps(eltype(scratch[k]))
    end
end
