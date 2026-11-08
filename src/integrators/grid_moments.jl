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
    grid_moments(f, grid, weights, Val(true); denominator)
end

# The mutating interface relies on its numeric input contract; the nonmutating
# interface retains validation. Both use the same centered reduction.
function grid_moments(f::F, grid, weights, ::Val{Check}; denominator=nothing) where {F, Check}
    first_positive = findfirst(w -> !iszero(w), weights)
    isnothing(first_positive) && throw(DomainError(weights, "Grid has zero mass"))
    w = float(weights[first_positive])
    Check && !(isfinite(w) && w > 0) &&
        throw(DomainError(w, "Weights must be finite and nonnegative"))
    origin = float(f(grid[first_positive]))
    Check && !(origin isa Real && isfinite(origin)) &&
        throw(DomainError(origin, "Samples must be finite real scalars"))
    mass = zero(w)
    moment = zero(w * origin)
    for i in eachindex(grid)
        wi = @inbounds weights[i]
        Check && !(isfinite(wi) && wi >= 0) &&
            throw(DomainError(wi, "Weights must be finite and nonnegative"))
        iszero(wi) && continue
        xi = i == first_positive ? origin : f(@inbounds grid[i])
        Check && !(xi isa Real && isfinite(xi)) &&
            throw(DomainError(xi, "Samples must be finite real scalars"))
        mass += wi
        moment += wi * (xi - origin)
    end
    mean, divisor = grid_moment_mean(mass, moment, origin, denominator)
    centered = zero(moment)
    for i in eachindex(grid)
        wi = @inbounds weights[i]
        iszero(wi) && continue
        centered += wi * (f(@inbounds grid[i]) - mean)^2
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

Return a tuple of `(mass, mean, variance)` summaries, one per weight buffer:

```julia
grid = [-1.0, 0.0, 1.0]
weights = [1.0, 2.0, 1.0]
buffers = (similar(weights), similar(weights))
summaries = weighted_grid_moments!(x -> (1 + x, 1 - x), grid, weights, buffers)
summaries[1] # (mass = 4.0, mean = 0.5, variance = 0.25)
```

Each buffer stores `weights[i] * factors(grid[i])[k]` in its element type.
`factors` supplies one factor per buffer, runs once per nonzero base weight,
and must not mutate inputs or buffers.

All arrays must have equal, nonempty axes. Supply at least one buffer; buffers
must not alias inputs or each other, including across concurrent calls. Factors,
base and produced weights must be finite and nonnegative, each channel's mass positive
and finite, and coordinates finite wherever produced weights are positive.
These per-point numeric requirements are caller obligations. `@inbounds` also
elides shape checks; aliasing, empty-input, mass and denominator checks remain.
Buffers may be partially written on failure.

Buffers retain produced weights for centered variance. The optional common
`denominator` follows [`weighted_grid_moments`](@ref). Warmed Float32/Float64 calls
allocate no heap storage; BigFloat arithmetic may allocate. Variance reductions
may reassociate floating-point additions.
"""
Base.@propagate_inbounds function weighted_grid_moments!(factors::F, grid::AbstractVector,
        weights::AbstractVector, buffers::NTuple{N, AbstractVector};
        denominator=nothing) where {F, N}
    materialized_grid_moments!(factors, grid, weights, buffers; denominator)
end

Base.@propagate_inbounds function materialized_grid_moments!(factors::F, grid,
        weights, buffers::NTuple{N, AbstractVector}; denominator=nothing) where {F, N}
    check_moment_buffers(grid, weights, buffers)
    for i in eachindex(grid)
        base = @inbounds weights[i]
        if iszero(base)
            foreach(buffer -> @inbounds(buffer[i] = zero(base)), buffers)
        else
            values = factors(@inbounds grid[i])
            @boundscheck length(values) == N || throw(DimensionMismatch("One factor is required per buffer"))
            for k in 1:N
                @inbounds buffers[k][i] = base * values[k]
            end
        end
    end
    map(buffer -> grid_moments(identity, grid, buffer, Val(false); denominator), buffers)
end

Base.@propagate_inbounds function check_moment_buffers(grid, weights, buffers::NTuple{N, AbstractVector}) where {N}
    N > 0 || throw(ArgumentError("At least one weight buffer is required"))
    isempty(weights) && throw(ArgumentError("Cannot summarize an empty grid"))
    @boundscheck axes(grid) == axes(weights) || throw(DimensionMismatch("Grid and weights must have equal axes"))
    for k in 1:N
        buffer = buffers[k]
        @boundscheck axes(buffer) == axes(grid) || throw(DimensionMismatch("Buffers and grid must have equal axes"))
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
    w = convert(eltype(buffer), base * factor)
    @inbounds buffer[i] = w
    iszero(w) && return state
    origin = iszero(state.mass) ? x : state.origin
    (; mass=state.mass + w, moment=state.moment + w * (x - origin), origin)
end

@inline function centered_grid_moment(grid::AbstractVector{T}, weights::AbstractVector{W},
        mean) where {T, W}
    total = zero(promote_type(T, W))
    @inbounds @simd for i in eachindex(grid)
        w = weights[i]
        # An unconditional addition lets LLVM recognize the reduction.
        total += iszero(w) ? zero(total) : w * (grid[i] - mean)^2
    end
    total
end

Base.@propagate_inbounds function weighted_grid_moments!(factors::F, grid::AbstractVector{T},
        weights::AbstractVector, buffers::Tuple{AbstractVector{W}, Vararg{AbstractVector{W}, N}};
        denominator=nothing) where {F, N, T <: AbstractFloat, W <: AbstractFloat}
    channels = N + 1
    if !isconcretetype(T) || !isconcretetype(W)
        return materialized_grid_moments!(factors, grid, weights, buffers; denominator)
    end
    check_moment_buffers(grid, weights, buffers)
    states = ntuple(_ -> (; mass=zero(W), moment=zero(promote_type(T, W)),
        origin=zero(T)), Val(channels))
    for i in eachindex(grid)
        base = @inbounds weights[i]
        if iszero(base)
            foreach(buffer -> @inbounds(buffer[i] = zero(W)), buffers)
        else
            x = @inbounds grid[i]
            values = factors(x)
            @boundscheck length(values) == channels || throw(DimensionMismatch("One factor is required per buffer"))
            factors_tuple = ntuple(k -> @inbounds(values[k]), Val(channels))
            states = map(buffers, factors_tuple, states) do buffer, factor, state
                @inline
                accumulate_grid_channel!(buffer, factor, state, base, x, i)
            end
        end
    end
    means_divisors = map(s -> grid_moment_mean(s.mass, s.moment, s.origin, denominator), states)
    map(states, means_divisors, buffers) do state, summary, buffer
        @inline
        total = centered_grid_moment(grid, buffer, summary[1])
        (; mass=state.mass, mean=summary[1], variance=total / summary[2])
    end
end
