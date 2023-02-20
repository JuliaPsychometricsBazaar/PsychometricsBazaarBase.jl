using PsychometricsBazzarBase
using Documenter

format = Documenter.HTML(
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://JuliaPsychometricsBazzar.github.io/PsychometricsBazzarBase.jl",
)

makedocs(;
    modules=[PsychometricsBazzarBase],
    authors="Frankie Robertson",
    repo="https://github.com/JuliaPsychometricsBazzar/PsychometricsBazzarBase.jl/blob/{commit}{path}#{line}",
    sitename="PsychometricsBazzarBase.jl",
    format=format,
    pages=[
        "PsychometricsBazzarBase.jl" => "index.md",
        "Integrators" => "integrators.md",
        "Optimizers" => "optimizers.md",
        "ConfigTools" => "config_tools.md",
        "IntegralCoeffs" => "integral_coeffs.md",
        "ExtraDistributions" => "extra_distributions.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaPsychometricsBazzar/PsychometricsBazzarBase.jl",
    devbranch="main",
)
