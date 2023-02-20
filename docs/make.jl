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
        "Getting started" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaPsychometricsBazzar/PsychometricsBazzarBase.jl",
    devbranch="main",
)
