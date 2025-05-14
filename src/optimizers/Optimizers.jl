"""
This module provides a common interface to different numerical optimization
techniques.
"""
module Optimizers

export FixedGridOptimizer, even_grid, quasimontecarlo_grid
export Optimizer, OneDimOptimOptimizer, MultiDimOptimOptimizer
export
# Optimization algorithms
## Zeroth order methods (heuristics)
      NelderMead,
      ParticleSwarm,
      SimulatedAnnealing,

## First order
### Quasi-Newton
      GradientDescent,
      BFGS,
      LBFGS,

### Conjugate gradient
      ConjugateGradient,

### Acceleration methods
      AcceleratedGradientDescent,
      MomentumGradientDescent,

### Nonlinear GMRES
      NGMRES,
      OACCEL,

## Second order
### (Quasi-)Newton
      Newton,

### Trust region
      NewtonTrustRegion,

# Constrained
## Box constraints, x_i in [lb_i, ub_i]
### Specifically Univariate, R -> R
      GoldenSection,
      Brent,

### Multivariate, R^N -> R
      Fminbox,
      SAMIN,

## Manifold constraints
      Manifold,
      Flat,
      Sphere,
      Stiefel,

## Non-linear constraints
      IPNewton

using ..ConfigTools
using ..Parameters

using Optim
using DocStringExtensions

abstract type Optimizer end
function Optimizer(bits...)
    @returnsome find1_instance(Optimizer, bits)
end

"""
Wraps an Optim.jl optimizer to optimize a single-dimensional domain function.

$(SIGNATURES)
"""
struct OneDimOptimOptimizer{OptimT <: Optim.AbstractOptimizer} <: Optimizer
    lo::Float64
    hi::Float64
    initial::Float64
    optim::OptimT
    opts::Optim.Options
end

function OneDimOptimOptimizer(lo, hi, optim)
    OneDimOptimOptimizer(lo, hi, lo + (hi - lo) / 2, optim, Optim.Options())
end

function (opt::OneDimOptimOptimizer{<: IPNewton})(
        f::F;
        lo = opt.lo,
        hi = opt.hi,
        initial = opt.initial,
        optim = opt.optim,
        opts = opt.opts
) where {F}
    df = TwiceDifferentiable(
        θ_arr -> -f(first(θ_arr)),
        [initial]
    )
    Optim.minimizer(optimize(
        df,
        TwiceDifferentiableConstraints([lo], [hi]),
        [initial],
        optim,
        opts
    ))[1]
end

function (opt::OneDimOptimOptimizer{<: NelderMead})(
        f::F;
        lo = opt.lo,
        hi = opt.hi,
        initial = opt.initial,
        optim = opt.optim,
        opts = opt.opts
) where {F}
    Optim.minimizer(optimize(
        θ_arr -> -f(first(θ_arr)),
        lo,
        hi,
        [initial],
        optim,
        opts
    ))[1]
end

"""
Wraps an Optim.jl optimizer to optimize a multi-dimensional domain function.

$(SIGNATURES)
"""
struct MultiDimOptimOptimizer{OptimT <: Optim.AbstractOptimizer} <: Optimizer
    lo::Vector{Float64}
    hi::Vector{Float64}
    initial::Vector{Float64}
    optim::OptimT
    opts::Optim.Options
end

function MultiDimOptimOptimizer(lo, hi, optim)
    MultiDimOptimOptimizer(lo, hi, lo + (hi - lo) / 2, optim, Optim.Options())
end

function (opt::MultiDimOptimOptimizer)(
        f::F;
        lo = opt.lo,
        hi = opt.hi,
        initial = opt.initial,
        optim = opt.optim,
        opts = opt.opts
) where {F}
    Optim.minimizer(optimize(
        θ_arr -> -f(θ_arr),
        lo,
        hi,
        initial,
        optim,
        opts
    ))
end

@kwdef struct NativeOneDimOptimOptimizer{T, MethodT <: Union{Brent,GoldenSection}} <: Optimizer
    lo::T
    hi::T
    method::MethodT=Brent()
    rel_tol::T=1e-4
    abs_tol::T=0.0
end

NativeOneDimOptimOptimizer(lo, hi) = NativeOneDimOptimOptimizer(lo, hi, Brent())

function (opt::NativeOneDimOptimOptimizer)(
        f::F;
        lo = opt.lo,
        hi = opt.hi,
        method = opt.method,
        rel_tol = opt.rel_tol,
        abs_tol = opt.abs_tol
) where {F}
    Optim.minimizer(optimize(
        θ -> -f(θ),
        lo,
        hi,
        method;
        rel_tol,
        abs_tol
    ))
end

include("./fixed.jl")

end
