import Cubature

"""
Construct a Cubature integrator based on `Cubature.jl` with on a specified interval.

$(TYPEDFIELDS)
"""
struct CubatureIntegrator{KwargsT} <: Integrator
    lo::Vector{Float64}
    hi::Vector{Float64}
    kwargs::KwargsT
end

function CubatureIntegrator(lo, hi; kwargs...)
    CubatureIntegrator(lo, hi, kwargs)
end

"""
    (integrator::CubatureIntegrator)(f[, ncomp, lo, hi; kwargs...])

Perform a Cubature integration based on `Cubature.jl`.
"""
function (integrator::CubatureIntegrator)(
    f::F,
    ncomp=1,
    lo=integrator.lo,
    hi=integrator.hi;
    kwargs...
) where F
    ErrorIntegrationResult(Cubature.hcubature(
        f, lo, hi;
        merge(integrator.kwargs, kwargs)...
    )...)
end
