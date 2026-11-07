export weighted_grid_moments

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
    isfinite(mass) || throw(DomainError(mass, "Grid mass must be finite"))
    divisor = isnothing(denominator) ? mass : denominator
    isfinite(divisor) && divisor > 0 || throw(DomainError(divisor, "Denominator must be positive and finite"))
    mean = origin * (mass / divisor) + moment / divisor
    centered = zero(moment)
    for i in eachindex(grid, weights)
        wi = weights[i]
        iszero(wi) && continue
        centered += wi * (f(grid[i]) - mean)^2
    end
    (; mass, mean, variance=centered / divisor)
end

weighted_grid_moments(grid, weights; kwargs...) =
    weighted_grid_moments(identity, grid, weights; kwargs...)

function weighted_grid_moments(f::F, grid, state::LogGridWeights; kwargs...) where {F}
    summary = weighted_grid_moments(f, grid, state.weights; kwargs...)
    (; mass=grid_mass(state), mean=summary.mean, variance=summary.variance,
        log_scale=state.log_scale, scaled_sum=state.scaled_sum)
end
