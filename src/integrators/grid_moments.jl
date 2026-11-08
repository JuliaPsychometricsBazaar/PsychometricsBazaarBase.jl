export weighted_grid_moments, weighted_grid_moments!

"""
    weighted_grid_moments([f,] grid, weights; denominator=nothing)
    weighted_grid_moments([f,] grid, state::LogGridWeights; denominator=nothing)

Return `(mass, mean, variance)` for scalar real samples `f(x)` (default `identity`).
Inputs are equally indexed vectors. Weights must be finite and nonnegative and
have positive finite total mass. Positive-weight samples must be finite; zero
weights never evaluate `f`. Empty/mismatched inputs and invalid values throw.

The first pass fuses mass and an origin-centered first moment, and the second
pass sums squared deviations from the mean. No cell width or Bessel correction
is applied. The warmed Float32/Float64 path needs no workspace or allocations.
BigFloat arithmetic retains input precision but may allocate.

By default both moments divide by the supplied weights' sum. An explicit positive
finite `denominator` replaces that divisor for *both* moments, including the mean
used to center the variance; `mass` still reports the actual weight sum.

For a log state, moments use normalized weights directly, while `mass` is a
logarithmic number. Additional `log_scale` and `scaled_sum` fields preserve the
separate normalization metadata. An explicit denominator is in normalized-weight
units in this overload. Vector means/covariance and their explicitly sized,
caller-owned workspaces are a future extension; this API only accepts scalars.
"""
function weighted_grid_moments(f::F, grid::AbstractVector, weights::AbstractVector;
        denominator=nothing) where {F}
    axes(grid) == axes(weights) || throw(DimensionMismatch("Grid and weights must have equal axes"))
    isempty(weights) && throw(ArgumentError("Cannot summarize an empty grid"))
    first_positive = findfirst(w -> !iszero(w), weights)
    isnothing(first_positive) && throw(DomainError(weights, "Grid has zero mass"))
    w = float(weights[first_positive])
    isfinite(w) && w > 0 || throw(DomainError(w, "Weights must be finite and nonnegative"))
    origin = float(f(grid[first_positive]))
    origin isa Real && isfinite(origin) || throw(DomainError(origin, "Samples must be finite real scalars"))
    mass = zero(w)
    moment = zero(w * origin)
    for i in eachindex(grid, weights)
        wi = weights[i]
        isfinite(wi) && wi >= 0 || throw(DomainError(wi, "Weights must be finite and nonnegative"))
        iszero(wi) && continue
        xi = i == first_positive ? origin : f(grid[i])
        xi isa Real && isfinite(xi) || throw(DomainError(xi, "Samples must be finite real scalars"))
        mass += wi
        moment += wi * (xi - origin)
    end
    mean, divisor = grid_moment_mean(mass, moment, origin, denominator)
    centered = zero(moment)
    for i in eachindex(grid, weights)
        wi = weights[i]
        iszero(wi) && continue
        centered += wi * (f(grid[i]) - mean)^2
    end
    (; mass, mean, variance=centered / divisor)
end

@inline function grid_moment_mean(mass, moment, origin, denominator)
    isfinite(mass) || throw(DomainError(mass, "Grid mass must be finite"))
    mass > 0 || throw(DomainError(mass, "Grid has zero mass"))
    divisor = isnothing(denominator) ? mass : denominator
    isfinite(divisor) && divisor > 0 || throw(DomainError(divisor, "Denominator must be positive and finite"))
    origin * (mass / divisor) + moment / divisor, divisor
end

weighted_grid_moments(grid, weights; kwargs...) =
    weighted_grid_moments(identity, grid, weights; kwargs...)

function weighted_grid_moments(f::F, grid, state::LogGridWeights; kwargs...) where {F}
    summary = weighted_grid_moments(f, grid, state.weights; kwargs...)
    (; mass=grid_mass(state), mean=summary.mean, variance=summary.variance,
        log_scale=state.log_scale, scaled_sum=state.scaled_sum)
end

"""
$(SIGNATURES)

Compute independent scalar moment summaries from transformed linear weights:

```julia
grid = [-1.0, 0.0, 1.0]
weights = [1.0, 2.0, 1.0]
buffers = (similar(weights), similar(weights))
summaries = weighted_grid_moments!(x -> (exp(x), exp(-x)), grid, weights, buffers)
```

`factors(x)` returns one finite nonnegative factor per buffer. It is called once
at each nonzero base weight, and never at zero base weights. Each buffer receives
`weights[i] * factors(grid[i])[k]`, converted to its element type. The returned
tuple contains `(mass, mean, variance)` for each channel. An optional common
`denominator` has the same meaning as in [`weighted_grid_moments`](@ref).

Floating-point grids and equally typed floating-point buffers fuse weight
production with mass/first-moment accumulation. A centered second pass reuses
the buffers without calling `factors` again. Other arrays use the same checked
summary after producing weights. Coordinates must be finite wherever a produced
weight is positive; each channel must have positive finite mass. No cell width
or Bessel correction is applied. This function consumes linear base weights;
it does not reconstruct absolute weights from log densities.

All arrays must have equal axes. Buffers must not alias inputs or each other;
use independent buffers for concurrent calls. Choose buffer types wide enough
for the products: there is no implicit precision policy or scratch allocation.
Buffers may be partially written if input validation or the callback throws.
The warmed, type-stable Float32/Float64 path allocates no heap storage; BigFloat
arithmetic may allocate. The callback must not mutate inputs or buffers.
"""
function weighted_grid_moments!(factors::F, grid::AbstractVector,
        weights::AbstractVector, buffers::NTuple{N, AbstractVector};
        denominator=nothing) where {F, N}
    check_moment_buffers(grid, weights, buffers)
    for i in eachindex(grid, weights)
        base = checked_grid_weight(weights[i])
        if iszero(base)
            foreach(buffer -> buffer[i] = zero(base), buffers)
        else
            values = factors(grid[i])
            length(values) == N || throw(DimensionMismatch("One factor is required per buffer"))
            for k in 1:N
                factor = checked_grid_weight(values[k])
                buffers[k][i] = base * factor
                checked_grid_weight(buffers[k][i])
            end
        end
    end
    map(buffer -> weighted_grid_moments(grid, buffer; denominator), buffers)
end

@inline function checked_grid_weight(w)
    isfinite(w) && w >= 0 || throw(DomainError(w, "Weights and factors must be finite and nonnegative"))
    w
end

function check_moment_buffers(grid, weights, buffers::NTuple{N, AbstractVector}) where {N}
    N > 0 || throw(ArgumentError("At least one weight buffer is required"))
    axes(grid) == axes(weights) || throw(DimensionMismatch("Grid and weights must have equal axes"))
    isempty(weights) && throw(ArgumentError("Cannot summarize an empty grid"))
    for k in 1:N
        buffer = buffers[k]
        axes(buffer) == axes(grid) || throw(DimensionMismatch("Buffers and grid must have equal axes"))
        (Base.mightalias(buffer, grid) || Base.mightalias(buffer, weights)) &&
            throw(ArgumentError("Weight buffers must not alias inputs"))
        for j in 1:(k - 1)
            Base.mightalias(buffer, buffers[j]) &&
                throw(ArgumentError("Weight buffers must not alias each other"))
        end
    end
    nothing
end

@inline function accumulate_grid_channel!(buffer, factor, state, base, x, i)
    checked_grid_weight(factor)
    w = convert(eltype(buffer), base * factor)
    checked_grid_weight(w)
    @inbounds buffer[i] = w
    iszero(w) && return state
    isfinite(x) || throw(DomainError(x, "Samples must be finite real scalars"))
    origin = iszero(state.mass) ? x : state.origin
    (; mass=state.mass + w, moment=state.moment + w * (x - origin), origin)
end

function weighted_grid_moments!(factors::F, grid::AbstractVector{T},
        weights::AbstractVector, buffers::Tuple{AbstractVector{W}, Vararg{AbstractVector{W}, N}};
        denominator=nothing) where {F, N, T <: AbstractFloat, W <: AbstractFloat}
    channels = N + 1
    if !isconcretetype(T) || !isconcretetype(W)
        return invoke(weighted_grid_moments!,
            Tuple{Any, AbstractVector, AbstractVector, NTuple{N + 1, AbstractVector}},
            factors, grid, weights, buffers; denominator)
    end
    check_moment_buffers(grid, weights, buffers)
    states = ntuple(_ -> (; mass=zero(W), moment=zero(promote_type(T, W)),
        origin=zero(T)), Val(channels))
    @inbounds for i in eachindex(grid, weights)
        base = checked_grid_weight(weights[i])
        if iszero(base)
            foreach(buffer -> buffer[i] = zero(W), buffers)
        else
            x = grid[i]
            values = factors(x)
            length(values) == channels || throw(DimensionMismatch("One factor is required per buffer"))
            factors_tuple = ntuple(k -> values[k], Val(channels))
            states = map(buffers, factors_tuple, states) do buffer, factor, state
                @inline
                accumulate_grid_channel!(buffer, factor, state, base, x, i)
            end
        end
    end
    means_divisors = map(s -> grid_moment_mean(s.mass, s.moment, s.origin, denominator), states)
    centered = ntuple(_ -> zero(promote_type(T, W)), Val(channels))
    @inbounds for i in eachindex(grid)
        centered = map(centered, buffers, means_divisors) do total, buffer, summary
            @inline
            w = @inbounds buffer[i]
            iszero(w) ? total : total + w * (@inbounds(grid[i]) - summary[1])^2
        end
    end
    map(states, means_divisors, centered) do state, summary, total
        (; mass=state.mass, mean=summary[1], variance=total / summary[2])
    end
end
