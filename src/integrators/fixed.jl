using QuasiMonteCarlo

struct FixedGridIntegrator{ContainerT <: Union{Vector{Float64}, Vector{Vector{Float64}}}} <:
       Integrator
    grid::ContainerT

    function FixedGridIntegrator(grid)
        new{Vector{Float64}}(grid)
    end

    function FixedGridIntegrator(grid::Vector{Vector{Float64}})
        new{Vector{Vector{Float64}}}(grid)
    end
end

function even_grid(theta_lo::Number, theta_hi::Number, quadpts)
    FixedGridIntegrator(range(theta_lo, theta_hi, quadpts))
end

function even_grid(theta_lo::AbstractVector, theta_hi::AbstractVector, quadpts_per_dim)
    prod = Iterators.product((
        range(lo, hi, length = quadpts_per_dim)
    for (lo, hi)
    in zip(theta_lo, theta_hi)
    )...)
    grid = reshape(collect.(prod), :)
    FixedGridIntegrator(grid)
end

function quasimontecarlo_grid(theta_lo, theta_hi, quadpts, sampler)
    grid = QuasiMonteCarlo.sample(quadpts, theta_lo, theta_hi, sampler)
    FixedGridIntegrator(grid)
end

function (integrator::FixedGridIntegrator)(args...; kwargs...)
    preallocate(integrator)(args...; kwargs...)
end

struct PreallocatedFixedGridIntegrator{ContainerT <:
                                       Union{Vector{Float64}, Matrix{Float64}}} <:
       Integrator
    inner::FixedGridIntegrator
    buf::ContainerT

    function PreallocatedFixedGridIntegrator(inner::FixedGridIntegrator{Vector{Float64}})
        quadpts = length(inner.grid)
        buf = Vector{Float64}(undef, quadpts)
        new{Vector{Float64}}(inner, buf)
    end

    function PreallocatedFixedGridIntegrator(inner::FixedGridIntegrator{Vector{Vector{Float64}}})
        quadpts = length(inner.grid)
        dim = length(inner.grid[1])
        buf = Matrix{Float64}(undef, quadpts, dim)
        new{Matrix{Float64}}(inner, buf)
    end
end

function (integrator::PreallocatedFixedGridIntegrator)(
        f::F,
        ncomp::Int = 0
) where {F}
    if ncomp == 0
        integrator.buf .= f.(integrator.inner.grid)
        BareIntegrationResult(sum(integrator.buf))
    else
        buf_rows = eachrow(integrator.buf)
        buf_rows .= f.(integrator.inner.grid)
        BareIntegrationResult(dropdims(sum(integrator.buf, dims = 1), dims = 1))
    end
end

function (integrator::PreallocatedFixedGridIntegrator)(
        f::F,
        init::AbstractVector{Float64},
        ncomp::Int = 0
) where {F}
    if ncomp == 0
        @. integrator.buf = (f.f)(integrator.inner.grid)
        @. integrator.buf = integrator.buf * init
        #@. integrator.buf = f(integrator.buf, integrator.inner.grid)
        BareIntegrationResult(sum(integrator.buf))
    else
        buf_rows = eachrow(integrator.buf)
        @. buf_rows = (f.f)(integrator.inner.grid)
        @. buf_rows = buf_rows * init
        BareIntegrationResult(dropdims(sum(integrator.buf, dims = 1), dims = 1))
    end
end

function preallocate(integrator::FixedGridIntegrator)
    PreallocatedFixedGridIntegrator(integrator)
end

function preallocate(integrator::Integrator)
    integrator
end

struct IterativeFixedGridIntegrator <: Integrator
    grid::Vector{Float64}
end

function (integrator::IterativeFixedGridIntegrator)(
        f::F,
        ncomp = 0
) where {F}
    if ncomp != 0
        error("IterativeFixedGridIntegrator only supports ncomp == 0")
    end
    s = 0.0
    for x in integrator.grid
        s += f(x)
    end
    BareIntegrationResult(s)
end
