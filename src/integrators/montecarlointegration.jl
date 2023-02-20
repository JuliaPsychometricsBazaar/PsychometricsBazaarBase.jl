using MonteCarloIntegration

"""
Construct a VEGAS integrator based on `MonteCarloIntegration.jl` with on a specified interval.
"""
struct MCIVegasIntegrator <: Integrator
    lo::Vector{Float64}
    hi::Vector{Float64}
    kwargs::NamedTuple
end

function MCIVegasIntegrator(lo, hi)
    MCIVegasIntegrator(lo, hi, ())
end

"""
Perform a VEGAS integration based on `MonteCarloIntegration.jl`.
"""
function (integrator::MCIVegasIntegrator)(
    f::F;
    ncomp=1,
    lo=integrator.lo,
    hi=integrator.hi,
    kwargs...
) where F
    ErrorIntegrationResult(vegas(
        f,
        lo,
        hi;
        merge(integrator.kwargs, kwargs)...
    )...)
end