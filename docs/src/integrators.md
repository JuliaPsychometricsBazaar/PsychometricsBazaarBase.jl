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
