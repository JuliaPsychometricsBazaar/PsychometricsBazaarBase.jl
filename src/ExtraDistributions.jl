"""
This module contains extra distributions for psychometrics. Currently this is
limited to `NormalScaledLogistic`.
"""
module ExtraDistributions

using Random: AbstractRNG
using Distributions: Logistic, ContinuousUnivariateDistribution, Normal, MvNormal, Zeros, ScalMat

using Lazy: @forward
using DocStringExtensions

export NormalScaledLogistic

# This seems to be the most commonly found exact value in the wild, see e.g. the
# R package `mirt``
const scaling_factor = 1.702

"""
The normal scaled logistic distribution is an approximation to the normal
distribution based upon the logistic distribution. It has been commonly used in
IRT modelling, such as in the `mirt` package for R.
"""
struct NormalScaledLogistic <: ContinuousUnivariateDistribution
    inner::Logistic
    """
    $(SIGNATURES)
    """
    NormalScaledLogistic(μ, σ) = Logistic(μ, σ / scaling_factor)
end

NormalScaledLogistic() = NormalScaledLogistic(0.0, 1.0)

@forward NormalScaledLogistic.inner (
    sampler, pdf, logpdf, cdf, quantile, minimum, maximum, insupport, mean, var,
    modes, mode, skewness, kurtosis, entropy, mgf, cf
)

end