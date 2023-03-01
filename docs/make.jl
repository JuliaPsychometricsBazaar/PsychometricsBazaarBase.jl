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
        "Home" => "index.md",
        "Modules" => ["integrators.md", "optimizers.md", "config_tools.md", "integral_coeffs.md", "const_distributions.md"]
    ],
)

deploydocs(;
    repo="github.com/JuliaPsychometricsBazzar/PsychometricsBazzarBase.jl",
    devbranch="main",
)
