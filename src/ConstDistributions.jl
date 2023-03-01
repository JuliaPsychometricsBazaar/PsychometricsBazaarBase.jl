"""
This module contains standard distributions for usage as transfer functions for IRT.
"""
module ConstDistributions

using Distributions: Logistic, Normal

using Lazy: @forward
using DocStringExtensions

export normal_scaled_logistic, std_normal

"""
This scaling facot seems to be the most commonly found exact value in the wild,
see e.g. the R package `mirt``
"""
const logistic_to_normal_scaling_factor = 1.702

"""
The normal scaled logistic distribution is an approximation to the normal
distribution based upon the logistic distribution. It has been commonly used in
IRT modelling, such as in the `mirt` package for R.
"""
const normal_scaled_logistic = Logistic(0.0, 1.0 / logistic_to_normal_scaling_factor)

"""
The standard normal distribution.
"""
const std_normal = Normal()

end
