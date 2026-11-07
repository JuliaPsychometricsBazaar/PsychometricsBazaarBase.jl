# Integrators

```@autodocs
Modules = [PsychometricsBazaarBase.Integrators]
```

## Index

```@index
Pages = ["integrators.md"]
```
## Log-density grid reductions

`LogGridWeights` normalizes log-density samples for an equal-weight grid. It is
independent of the integration backend and does not add a cell-width factor:

```julia
using PsychometricsBazaarBase.Integrators

grid = [-1.0, 0.0, 1.0]
state = LogGridWeights([-1000.0, -1001.0, -1002.0])
mean = weighted_grid_mean(identity, grid, state)
mass = grid_mass(state) # logarithmic number; does not underflow
integral = scale_grid_value(mean, state) # signed logarithmic number
```

Coefficient functions can return scalars, vectors or matrices. Normalized moments
are computed directly from weights; do not recover them by dividing raw integrals
or subtracting large logarithms. Points with zero normalized weight skip the
coefficient function. For reusable buffers, `normalize_grid_weights!` returns a
new state sharing the supplied log and weight arrays.

## Scalar weighted moments

`weighted_grid_moments` consumes evaluated weights without evaluating a density:

```julia
summary = weighted_grid_moments([1e8 - 1, 1e8 + 1], [1.0, 1.0])
# (mass = 2.0, mean = 1e8, variance = 1.0)
summary = weighted_grid_moments(grid, state)
# Moments use state.weights; mass is a logarithmic number.
```

Mass and an origin-centered first moment share a pass, followed by a pass over
squared deviations. This avoids cancellation from subtracting raw second moments.
A coefficient overload `weighted_grid_moments(f, grid, weights)` skips `f` at
zero-weight points. Coefficients must be pure scalar functions, since they are
called in both passes. The warmed Float32/Float64 path allocates no scratch;
each caller supplies its own input arrays. BigFloat arithmetic may allocate.
Vector means and covariance with explicitly sized caller-owned workspaces remain
a follow-up.

An explicit `denominator` divides both moment sums and determines the mean used
to center deviations; it does not change the returned actual mass. For log states
it is expressed in normalized-weight units. Without an explicit denominator,
positive weight rescaling and common log-density shifts leave moments invariant.
Log-state results retain `log_scale` and `scaled_sum` separately; no tiny absolute
density is exponentiated for the normalized moments. All paths retain sum
semantics, with no implicit spacing factor or Bessel correction.

Run `julia --project=. benchmark/grid_moments.jl` for warmed 33-, 65-, and
129-point reductions, reporting setup cost separately from repeated calls.
