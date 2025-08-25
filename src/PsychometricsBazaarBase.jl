module PsychometricsBazaarBase

using DocStringExtensions

public Parameters, ConfigTools, IntegralCoeffs, Integrators, ConstDistributions,
       Interpolators, Optimizers

function show_into_string(obj, mime::MIME = MIME("text/plain"))
    buf = IOBuffer()
    show(buf, mime, rules)
    return String(take!(buf))
end

function show_into_buf(obj, mime::MIME = MIME("text/plain"))
    buf = IOBuffer()
    show(buf, mime, rules)
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
