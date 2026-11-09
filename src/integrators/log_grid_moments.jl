export weighted_log_grid_moments!

"""
    weighted_log_grid_moments!(log_factors, grid, logs, log_buffers, weight_buffers;
                              skip_checks=false)

```julia
grid = [-1.0, 0.0, 1.0]
logs = [-1001.0, -1000.0, -1001.0]
scratch = (similar(logs), similar(logs))
weights = (similar(logs), similar(logs))
weighted_log_grid_moments!(x -> (-x^2, -2x^2), grid, logs, scratch, weights)
```

Return one `(log_scale, scaled_sum, mean, variance)` summary per channel.
Each channel adds its log factor to `logs`, subtracts its maximum, then computes
origin-centered means and centered variances from the resulting scaled weights.
The original mass is `exp(log_scale) * scaled_sum`; it need not be representable.
Weight buffers retain **scaled**, not unit-sum, weights.

The callback runs once per grid point, in order, and returns one log factor per
channel. Inputs and produced logs must be finite or `-Inf`, coordinates finite,
and each channel must have positive mass. Arrays must be nonempty, equally
indexed, and all output buffers disjoint from inputs and each other. Numeric
validity is a caller obligation. `skip_checks=true` also assumes shape, aliasing
and factor-count contracts; zero-mass channel checks always remain. Buffers may
be partially written on failure. Reductions may reassociate additions.
"""
Base.@propagate_inbounds function weighted_log_grid_moments!(log_factors::F,
        grid::AbstractVector, logs::AbstractVector,
        log_buffers::NTuple{N, AbstractVector}, weight_buffers::NTuple{N, AbstractVector};
        skip_checks::Bool=false) where {F, N}
    skip_checks || check_moment_buffers(grid, logs, (log_buffers..., weight_buffers...))
    shifts = materialize_grid_logs!(log_factors, grid, logs, log_buffers, skip_checks)
    map(shifts) do shift
        isfinite(shift) || throw(DomainError(shift, "Channel must have positive finite scaled mass"))
    end
    scaled_log_grid_moments!(grid, log_buffers, weight_buffers, shifts)
end

Base.@propagate_inbounds function materialize_grid_logs!(factors::F, grid, logs,
        buffers::NTuple{N, AbstractVector}, skip_checks::Bool) where {F, N}
    shifts = map(buffer -> oftype(float(zero(eltype(buffer))), -Inf), buffers)
    for i in eachindex(grid)
        base = @inbounds logs[i]
        values = @inline factors(@inbounds grid[i])
        if !skip_checks
            @boundscheck length(values) == N || throw(DimensionMismatch("One log factor is required per buffer"))
        end
        produced = map(ntuple(identity, Val(N))) do k
            value = base + values[k]
            @inbounds buffers[k][i] = value
            value
        end
        shifts = map(max, shifts, produced)
    end
    shifts
end

@inline function scaled_log_grid_moments!(grid, logs::NTuple{N, AbstractVector},
        buffers::NTuple{N, AbstractVector}, shifts) where {N}
    i = first(eachindex(grid))
    origin = float(@inbounds grid[i])
    masses = map(buffer -> zero(float(eltype(buffer))), buffers)
    moments = map(mass -> zero(mass * origin), masses)
    @inbounds @simd for i in eachindex(grid)
        delta = grid[i] - origin
        weights = map(logs, buffers, shifts) do log_values, buffer, shift
            w = @inline exp(log_values[i] - shift)
            buffer[i] = w
            w
        end
        masses = map(+, masses, weights)
        moments = map((total, w) -> total + w * delta, moments, weights)
    end
    summaries = map((mass, moment) -> grid_moment_mean(mass, moment, origin, nothing), masses, moments)
    centered = centered_grid_moments(grid, buffers, map(first, summaries))
    map((shift, mass, summary, total) ->
        (; log_scale=shift, scaled_sum=mass, mean=summary[1], variance=total / summary[2]),
        shifts, masses, summaries, centered)
end
