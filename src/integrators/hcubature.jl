import HCubature

"""
Construct a Cubature integrator based on `HCubature.jl` with on a specified interval.

$(TYPEDFIELDS)
"""
struct HCubatureIntegrator{KwargsT} <: Integrator
    lo::Vector{Float64}
    hi::Vector{Float64}
    kwargs::KwargsT
end

function HCubatureIntegrator(lo, hi; kwargs...)
    HCubatureIntegrator(lo, hi, kwargs)
end

"""
    (integrator::HCubatureIntegrator)(f[, ncomp, lo, hi; kwargs...])

Perform Cubature integration based on `HCubature.jl`.
"""
function (integrator::HCubatureIntegrator)(
        f::F,
        ncomp = 1;
        lo = integrator.lo,
        hi = integrator.hi,
        kwargs...
) where {F}
    ErrorIntegrationResult(HCubature.hcubature(
        f, lo, hi;
        merge(integrator.kwargs, kwargs)...
    )...)
end
