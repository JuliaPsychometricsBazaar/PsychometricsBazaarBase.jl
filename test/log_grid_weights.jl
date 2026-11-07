using PsychometricsBazaarBase.Integrators

@testset "stable grid normalization" begin
    for logs in ([0.0, 0.0, 0.0], fill(-1e300, 3))
        state = LogGridWeights(logs)
        @test state.weights ≈ fill(1 / 3, 3)
        @test weighted_grid_mean(identity, [-1, 0, 2], state) ≈ 1 / 3
        @test state.scaled_sum == 3
    end
    state = LogGridWeights([-1000.0, -1001, -Inf])
    f(x) = x == 99 ? error("outside support") : x
    @test weighted_grid_mean(f, [-2, 1, 99], state) ≈ (-2 + exp(-1)) / (1 + exp(-1))
    @test !iszero(grid_mass(state))
    @test log(grid_mass(state)) ≈ -1000 + log1p(exp(-1))
    value = scale_grid_value(weighted_grid_mean(f, [-2, 1, 99], state), state)
    @test value < 0
    @test log(abs(value)) ≈ -1000 + log(2 - exp(-1))
    @test Float64(grid_mass(state)) == 0
    for f in (x -> [x, x^2], x -> [x -x^2; -x^2 -x])
        expected = (f(-2) + exp(-1) * f(1)) / (1 + exp(-1))
        @test weighted_grid_mean(f, [-2, 1, 99], state) ≈ expected
        @test Float64.(scale_grid_value(expected, LogGridWeights([0.0, -1, -Inf]))) ≈ expected * (1 + exp(-1))
    end
    covariance = weighted_grid_mean(x -> [x^2 -x^2; -x^2 x^2],
        [-1, 1], LogGridWeights([-1000., -1000.]))
    @test covariance == [1 -1; -1 1]
    @test eltype(LogGridWeights(Float32[0, -1]).weights) == Float32
    @test weighted_grid_mean(identity, [-1, 1], LogGridWeights([-1000., -1000.])) == 0
    @test iszero(scale_grid_value(0.0, state))
    setprecision(256) do
        logs = BigFloat[-1000, -1001, -1002]
        state = LogGridWeights(logs)
        @test LogGridWeights(logs .- big"1e70").weights ≈ state.weights rtol=big"1e-70"
        reference = exp.(logs) / sum(exp.(logs))
        @test eltype(state.weights) == BigFloat
        @test state.weights ≈ reference rtol=big"1e-70"
        @test weighted_grid_mean(identity, BigFloat[-2, 1, 3], state) ≈ sum(reference .* [-2, 1, 3]) rtol=big"1e-70"
    end
end

@testset "validation and buffer reuse" begin
    @test_throws ArgumentError LogGridWeights(Float64[])
    for logs in ([-Inf, -Inf], [NaN, 0.0], [Inf, 0.0])
        @test_throws DomainError LogGridWeights(logs)
        @test_throws DomainError normalize_grid_weights!(logs, similar(logs))
    end
    @test_throws ArgumentError normalize_grid_weights!(Float64[], Float64[])
    @test_throws DimensionMismatch normalize_grid_weights!([0.0], zeros(2))
    logs = [-1.0, -2.0]
    @test_throws ArgumentError normalize_grid_weights!(logs, logs)
    weights = similar(logs)
    state = normalize_grid_weights!(logs, weights)
    @test state.weights === weights
    @test state.log_values === logs
    @test weights ≈ LogGridWeights(logs).weights
    logs .= [-3, -3]
    state = normalize_grid_weights!(logs, weights)
    @test state.weights == [0.5, 0.5]
    @test state.log_scale == -3
    @test_throws DimensionMismatch weighted_grid_mean(identity, [1], state)
    @test_throws ArgumentError weighted_grid_sum(identity, [1, 2], zeros(2))
end
