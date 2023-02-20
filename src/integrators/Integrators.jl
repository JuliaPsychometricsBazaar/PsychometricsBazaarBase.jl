"""
This module provides a common interface to different numerical integration techniques.
"""
module Integrators

export Integrator, QuadGKIntegrator, FixedGKIntegrator
export CubatureIntegrator, HCubatureIntegrator, MultiDimFixedGKIntegrator
export normdenom, intval, interr
export CubaVegas, CubaSuave, CubaDivonne, CubaCuhre
export IntReturnType, IntValue, IntMeasurement
export CubaIntegrator, CubaVegas, CubaSuave, CubaDivonne, CubaCuhre

using ..ConfigTools
using ..IntegralCoeffs: one
import Measurements

abstract type Integrator end
function Integrator(bits...)
    @returnsome find1_instance(Integrator, bits)
end

abstract type IntReturnType end
struct IntValue <: IntReturnType end
struct IntMeasurement <: IntReturnType end

function (::IntValue)(res)
    intval(res)
end

function (::IntMeasurement)(res)
    intmes(res)
end

function normdenom(integrator::Integrator; options...)
    normdenom(IntValue(), integrator; options...)
end

function normdenom(rett::IntReturnType, integrator::Integrator; lo=integrator.lo, hi=integrator.hi, options...)
    # XXX: Presumably we can just return the analytic value here instead? Is this function even needed?
    rett(integrator(one; lo=lo, hi=hi, options...))
end

struct ScaleUnitDomain{F}
    f::F
    lo::Vector{Float64}
    interval::Vector{Float64}
    scaler::Float64

    function ScaleUnitDomain(f::F, lo, hi) where {F}
        interval = hi .- lo
        new{F}(
            f,
            lo,
            interval,
            prod(interval)
        )
    end
end

function (sud::ScaleUnitDomain)(x)
    sud.scaler * sud.f(sud.lo .+ sud.interval .* x)
end

# Values from fscore() from mirt/mirtCAT
function mirtcat_quadpnts(nd)
    if nd == 1
        61
    elseif nd == 2
        31
    elseif nd == 3
        15
    elseif nd == 4
        9
    elseif nd == 5
        7
    else
        3
    end
end

"""
The result of an integration technique which provides no error value.
"""
struct BareIntegrationResult{VecT}
    vec::VecT
end

"""
The result of an integration technique which provides an error value. Note that
error values are not comparible between different integration techniques in
general.
"""
struct ErrorIntegrationResult{VecT, ErrT}
    vec::VecT
    err::ErrT
end

"""
Given any integration result, get the integral value.
"""
function intval(res::Union{BareIntegrationResult, ErrorIntegrationResult})
    res.vec
end

"""
Given any integration result, get the integral error. In case the integration
technique does not supply one, this returns `nothing`.
"""
function interr end

function interr(::BareIntegrationResult)
    nothing
end

function interr(res::ErrorIntegrationResult)
    res.err
end

function intmes(res::ErrorIntegrationResult)
    Measurements.measurement(intval(res), interr(res))
end

include("./quadgk.jl")
include("./hcubature.jl")
include("./montecarlointegration.jl")
include("./cubature.jl")
include("./cuba.jl")

end