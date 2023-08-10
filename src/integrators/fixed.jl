struct FixedGridIntegrator <: Integrator
    grid::Vector{Float64}
end

function even_grid(theta_lo, theta_hi, quadpts)
    FixedGridIntegrator(range(theta_lo, theta_hi, quadpts))
end

function quasimontecarlo_grid(theta_lo, theta_hi, quadpts, sampler)
    FixedGridIntegrator(QuasiMonteCarlo.sample(quadpts, theta_lo, theta_hi, sampler))
end

function (integrator::FixedGridIntegrator)(
    f::F,
    ncomp=0
) where F
    preallocate(integrator)(f, ncomp)
end

struct PreallocatedFixedGridIntegrator <: Integrator
    inner::FixedGridIntegrator
    buf::Vector{Float64}

    PreallocatedFixedGridIntegrator(inner::FixedGridIntegrator) = new(inner, Vector{Float64}(undef, length(inner.grid)))
end

function (integrator::PreallocatedFixedGridIntegrator)(
    f::F,
    ncomp=0
) where F
    if ncomp != 0
        error("FixedGridIntegrator only supports ncomp == 0")
    end
    integrator.buf .= f.(integrator.inner.grid)
    BareIntegrationResult(sum(integrator.buf))
end

function preallocate(integrator::FixedGridIntegrator)
    PreallocatedFixedGridIntegrator(integrator)
end

function preallocate(integrator::Integrator)
    integrator
end