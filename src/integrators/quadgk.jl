using QuadGK
using QuadGK: cachedrule, evalrule, Segment
using LinearAlgebra: norm
using FillArrays
import Base.Iterators

function fixed_gk(f::F, lo, hi, n) where {F}
    x, w, gw = cachedrule(Float64, n)

    seg = evalrule(f, lo, hi, x, w, gw, norm)
    (seg.I, seg.E)
end

"""
Construct a adaptive Gauss-Kronrod integrator based on `QuadGK.jl` with on a specified interval.

$(TYPEDFIELDS)
"""
struct QuadGKIntegrator <: Integrator
    lo::Float64
    hi::Float64
    order::Int
end

# This could be unsafe if quadgk performed i/o. It might be wise to switch to
# explicitly passing this through from the caller at some point.
# Just preallocate an arbitrary size for now (easiest, would make more sense to use 'order' somehow but we don't have it)
# It's 24 * 100 * threads bytes, ~10kb for 4 threads which is unconditionally allocated when this library is used
segbufs = [Vector{Segment{Float64, Float64, Float64}}(undef, 100) for _ in Threads.nthreads()]

"""
    (integrator::QuadGKIntegrator)(f[, ncomp, lo, hi; order=..., rtol=...])

Perform an adaptive Gauss-Kronrod integration using `QuadGK.jl`.
"""
function (integrator::QuadGKIntegrator)(
    f::F,
    ncomp=0,
    lo=integrator.lo,
    hi=integrator.hi;
    order=integrator.order,
    rtol=1e-4
) where F
    if ncomp != 0
        error("QuadGKIntegrator only supports ncomp == 0")
    end
    ErrorIntegrationResult(quadgk(f, lo, hi, rtol=rtol, segbuf=segbufs[Threads.threadid()], order=order)...)
end

"""
Construct a fixed-order Gauss-Kronrod integrator based on `QuadGK.jl` with on a specified interval.

$(TYPEDFIELDS)
"""
struct FixedGKIntegrator <: Integrator
    lo::Float64
    hi::Float64
    order::Int
end

"""
    (integrator::QuadGKIntegrator)(f[, ncomp, lo, hi; order=...])

Perform a fixed-order Gauss-Kronrod integration based on `QuadGK.jl`.
"""
function (integrator::FixedGKIntegrator)(
    f::F,
    ncomp=0,
    lo=integrator.lo,
    hi=integrator.hi;
    order=integrator.order
) where F
    if ncomp != 0
        error("FixedGKIntegrator only supports ncomp == 0")
    end
    ErrorIntegrationResult(fixed_gk(f, lo, hi, order)...)
end

"""
Construct a fixed-order multi-dimensional Gauss-Kronrod integrator based on
`QuadGK.jl` with on a specified interval.

$(TYPEDFIELDS)
"""
struct MultiDimFixedGKIntegrator{OrderT <: AbstractVector{Int}} <: Integrator
    lo::Vector{Float64}
    hi::Vector{Float64}
    order::OrderT
end

function MultiDimFixedGKIntegrator(lo, hi)
    MultiDimFixedGKIntegrator(lo, hi, mirtcat_quadpnts(length(lo)))
end

function MultiDimFixedGKIntegrator(lo, hi, order::Int)
    MultiDimFixedGKIntegrator(lo, hi, Fill(order, length(lo)))
end

"""
    (integrator::QuadGKIntegrator)(f[, ncomp, lo, hi; order=...])

Perform a fixed-order multi-dimensional Gauss-Kronrod integrator based on `QuadGK.jl`.
"""
function (integrator::MultiDimFixedGKIntegrator)(
    f::F,
    ncomp=1,
    lo=integrator.lo,
    hi=integrator.hi;
    order=integrator.order
) where F
    x = Array{Float64}(undef, length(lo))
    function inner(idx)
        function integrate()
            return fixed_gk(inner(idx + 1), lo[idx + 1], hi[idx + 1], order[idx + 1])[1]
        end
        function f1d(x1d)
            x[idx] = x1d
            if idx >= length(lo)
                #@info "Calling f" x
                return f(x)
            else
                return integrate()
            end
        end
        if idx == 0
            return integrate()
        else
            return f1d
        end
    end
    # TODO: Combine errors somehow
    BareIntegrationResult(inner(0))
end
