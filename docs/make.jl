using PsychometricsBazaarBase
using Documenter

format = Documenter.HTML(
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://JuliaPsychometricsBazaar.github.io/PsychometricsBazaarBase.jl",
)

makedocs(;
    modules=[PsychometricsBazaarBase],
    authors="Frankie Robertson",
    repo="https://github.com/JuliaPsychometricsBazaar/PsychometricsBazaarBase.jl/blob/{commit}{path}#{line}",
    sitename="PsychometricsBazaarBase.jl",
    format=format,
    pages=[
        "Home" => "index.md",
        "Modules" => ["integrators.md", "optimizers.md", "config_tools.md", "integral_coeffs.md", "const_distributions.md"]
    ],
    warnonly = [:missing_docs],
)

deploydocs(;
    repo="github.com/JuliaPsychometricsBazaar/PsychometricsBazaarBase.jl",
    devbranch="main",
)
