using LogarithmicNumbers: Logarithmic, ULogarithmic

export LogGridWeights, normalize_grid_weights!, weighted_grid_mean, weighted_grid_sum
export grid_mass, scale_grid_value

"""
    LogGridWeights(log_values::AbstractVector)

Normalize equal-weight grid log densities without exponentiating their common
scale. Stores owned `log_values`, normalized `weights`, `log_scale` (the maximum),
and `scaled_sum` separately to preserve normalization under huge common shifts.
Finite real values and `-Inf` are accepted; empty and zero-mass grids are errors.
Floating-point sample types, including BigFloat, are preserved. Treat the stored
arrays as read-only; use [`normalize_grid_weights!`](@ref) to refresh a state.
"""
struct LogGridWeights{T <: AbstractFloat}
    log_values::Vector{T}
    weights::Vector{T}
    log_scale::T
    scaled_sum::T
end

function LogGridWeights(log_values::AbstractVector)
    logs = float.(collect(log_values))
    normalize_grid_weights!(logs, similar(logs))
end

"""
    normalize_grid_weights!(logs::Vector{T}, weights::Vector{T}) where T <: AbstractFloat

Normalize into an equally sized, non-aliasing buffer. The returned state shares
both arrays; its scalar fields replace those in any previous state. Do not use
an older state after reusing its buffers.
"""
function normalize_grid_weights!(logs::Vector{T}, weights::Vector{T}) where {T <: AbstractFloat}
    isempty(logs) && throw(ArgumentError("Cannot normalize an empty grid"))
    length(logs) == length(weights) ||
        throw(DimensionMismatch("Log densities and weights differ in length"))
    Base.mightalias(logs, weights) &&
        throw(ArgumentError("Log densities and weight buffers must not alias"))
    all(x -> isfinite(x) || x == -Inf, logs) ||
        throw(DomainError(logs, "Grid log densities must be finite or -Inf"))
    shift = maximum(logs)
    isfinite(shift) ||
        throw(DomainError(shift, "The density has zero mass on the entire grid"))
    weights .= exp.(logs .- shift)
    total = sum(weights)
    weights ./= total
    LogGridWeights(logs, weights, shift, total)
end

# Do not form exp(log_values - log_normalizer): adding log(scaled_sum) to
# an extremely large-magnitude shift can lose all the normalization precision.
"""
    weighted_grid_mean(f, grid, state::LogGridWeights)

Evaluate a normalized expectation directly, retaining signed scalar, vector or
matrix coefficients. Zero-weight points do not evaluate `f`.
"""
function weighted_grid_mean(f::F, grid, state::LogGridWeights) where {F}
    weighted_grid_sum(f, grid, state.weights)
end

"""
    weighted_grid_sum(f, grid, weights::AbstractVector)

Sum `w * f(x)` over equal-length grid and weight collections, skipping zero
weights before calling `f`. At least one weight must be nonzero.
"""
function weighted_grid_sum(f::F, grid, weights::AbstractVector) where {F}
    length(grid) == length(weights) ||
        throw(DimensionMismatch("Grid and weights differ in length"))
    any(!iszero, weights) || throw(ArgumentError("All grid weights are zero"))
    # Zero-mass points may lie outside f's domain. They contribute nothing.
    sum(w * f(x) for (x, w) in zip(grid, weights) if !iszero(w))
end

"""
    grid_mass(state::LogGridWeights)

Return the original unnormalized sum as an unsigned logarithmic number, avoiding
Float64 underflow. No cell-width factor is included.
"""
grid_mass(state::LogGridWeights) = exp(ULogarithmic, state.log_scale) * state.scaled_sum

"""
    scale_grid_value(value, state::LogGridWeights)

Multiply a normalized scalar or array expectation by the original mass, returning
signed logarithmic numbers. For an integral use
`scale_grid_value(weighted_grid_mean(f, grid, state), state)`.
"""
function scale_grid_value(value::Number, state::LogGridWeights)
    Logarithmic(value) * grid_mass(state)
end

function scale_grid_value(value::AbstractArray, state::LogGridWeights)
    map(x -> scale_grid_value(x, state), value)
end
