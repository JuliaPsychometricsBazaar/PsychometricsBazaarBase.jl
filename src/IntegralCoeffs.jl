"""
These are helpers for weighting integrals of p.d.f.s for calculating basic
stats.

The main idea of doing it this way is to have a single instance of these to
reuse specializations and to use structs so as to be able to control the
level of specialization.
"""
module IntegralCoeffs

using Distributions: Distribution, pdf
using DocStringExtensions

"""
$(SIGNATURES)
"""
@inline function one(x_)::Float64
    1.0
end

"""
$(SIGNATURES)
"""
@inline function id(x::T)::T where {T}
    x
end

"""
$(SIGNATURES)
"""
struct SqDev{CenterT}
    center::CenterT
end

@inline function (sq_dev::SqDev{CenterT})(x::CenterT)::CenterT where {CenterT}
    (x .- sq_dev.center) .^ 2
end

struct Prior{Dist <: Distribution}
    dist::Dist
end

"""
$(SIGNATURES)
"""
struct PriorApply{Dist, F}
    prior::Prior{Dist}
    func::F
end

@inline function (prior_apply::PriorApply)(x)
    pdf(prior_apply.prior.dist, x) * prior_apply.func(x)
end

end
