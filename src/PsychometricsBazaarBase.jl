module PsychometricsBazaarBase

using DocStringExtensions

public Parameters, ConfigTools, IntegralCoeffs, Integrators, ConstDistributions,
       Interpolators, Optimizers

export power_summary, show_into_string, show_into_buf, power_summary_into_string,
       power_summary_into_buf

"""
power_summary(io::IO, obj::Any; kwargs...)

Display a summary of `obj` suitable for power users, avoiding using internal Julia types.

All arguments are passed by kwargs which may vary by the type of `obj`.

In practice, show(::IO, MIME"text/plain", obj) may use this internally.
"""
function power_summary end

function show_into_string(obj, mime::MIME = MIME("text/plain"); kwargs...)
    return String(take!(show_into_buf(obj, mime; kwargs...)))
end

function show_into_buf(obj, mime::MIME = MIME("text/plain"); kwargs...)
    buf = IOBuffer()
    show(buf, mime, obj; kwargs...)
    return seekstart(buf)
end

function power_summary_into_string(obj; kwargs...)
    return String(take!(power_summary_into_buf(obj; kwargs...)))
end

function power_summary_into_buf(obj, mime::MIME = MIME("text/plain"); kwargs...)
    buf = IOBuffer()
    power_summary(buf, obj; kwargs...)
    return seekstart(buf)
end

include("./vendor/IndentWrappers.jl")
include("./vendor/Parameters.jl")
include("./ConfigTools.jl")
include("./IntegralCoeffs.jl")
include("./integrators/Integrators.jl")
include("./ConstDistributions.jl")
include("./Interpolators.jl")
include("./optimizers/Optimizers.jl")

end
