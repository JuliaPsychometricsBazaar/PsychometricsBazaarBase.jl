using QuasiMonteCarlo

struct FixedGridIntegrator{ContainerT <: AbstractVector} <:
       Integrator
    grid::ContainerT
end

function even_grid(theta_lo::Number, theta_hi::Number, quadpts)
    FixedGridIntegrator(range(theta_lo, theta_hi, quadpts))
end

function even_grid(theta_lo::AbstractVector, theta_hi::AbstractVector,
        quadpts_per_dim; impl = FixedGridIntegrator)
    prod = Iterators.product((
        range(lo, hi, length = quadpts_per_dim)
    for (lo, hi)
    in zip(theta_lo, theta_hi)
    )...)
    grid = reshape(collect.(prod), :)
    impl(grid)
end

function quasimontecarlo_grid(
        theta_lo, theta_hi, quadpts, sampler; impl = FixedGridIntegrator)
    grid = QuasiMonteCarlo.sample(quadpts, theta_lo, theta_hi, sampler)
    impl(grid)
end

function (integrator::FixedGridIntegrator)(args...; kwargs...)
    preallocate(integrator)(args...; kwargs...)
end

struct PreallocatedFixedGridIntegrator{ContainerT <: AbstractArray,
                                       FixedGridIntegratorT <: FixedGridIntegrator} <:
       Integrator
    inner::FixedGridIntegratorT
    buf::ContainerT

    function PreallocatedFixedGridIntegrator(inner::FixedGridIntegrator{<:AbstractVector{<:Number}})
        quadpts = length(inner.grid)
        buf = Vector{float(eltype(inner.grid))}(undef, quadpts)
        new{typeof(buf), typeof(inner)}(inner, buf)
    end

    function PreallocatedFixedGridIntegrator(inner::FixedGridIntegrator{<:AbstractVector{<:AbstractVector}})
        quadpts = length(inner.grid)
        # XXX: In general this is wrong. The output dimension could be anything.
        # TODO: Instead it be that we have a maximum output dimension specified at construction time
        dim = length(inner.grid[1])
        buf = Matrix{float(eltype(eltype(inner.grid)))}(undef, quadpts, dim)
        new{typeof(buf), typeof(inner)}(inner, buf)
    end
end

function (integrator::PreallocatedFixedGridIntegrator{<:AbstractVector})(
    f::F,
    ncomp::Int = 0
) where {F}
    if ncomp == 0 || ncomp == 1
        integrator.buf .= f.(integrator.inner.grid)
        BareIntegrationResult(sum(integrator.buf))
    else
        error("ncomp must be 0 or 1 for FixedGridIntegrator with a vector buffer")
    end
end

function (integrator::PreallocatedFixedGridIntegrator{<:AbstractMatrix})(
    f::F,
    ncomp::Int = 0
) where {F}
    if ncomp == 0 || ncomp == 1
        integrator.buf[:, 1] .= f.(integrator.inner.grid)
        BareIntegrationResult(sum(@view integrator.buf[:, 1]))
    else
        buf_rows = eachrow(integrator.buf)
        buf_rows .= f.(integrator.inner.grid)
        BareIntegrationResult(dropdims(sum(integrator.buf, dims = 1), dims = 1))
    end
end

function (integrator::PreallocatedFixedGridIntegrator)(
        f::F,
        init::AbstractVector,
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

struct IterativeFixedGridIntegrator{ContainerT <: AbstractVector} <:
       Integrator
    grid::ContainerT
end


function (integrator::IterativeFixedGridIntegrator)(f::F, ncomp = nothing) where {F}
    s = sum(f, integrator.grid)
    BareIntegrationResult(s)
end

function power_summary(io::IO, integrator::Union{FixedGridIntegrator, IterativeFixedGridIntegrator})
    if integrator.grid isa AbstractRange
        println(io, "Fixed step grid integrator:")
    else
        println(io, "Array-based grid integrator:")
    end
    power_summary(indent(io, 2), GridSummary(integrator.grid))
end

show(io::IO, ::MIME"text/plain", integrator::PreallocatedFixedGridIntegrator) = show(io, MIME("text/plain"), integrator)

struct MidpointIntegrator{XsT <: AbstractVector, BufT <: AbstractVector} <: Integrator
    xs::XsT
    buf::BufT

    function MidpointIntegrator(xs)
        buf = Vector{float(eltype(xs))}(undef, length(xs))
        new{typeof(xs), typeof(buf)}(xs, buf)
    end
end

function (integrator::MidpointIntegrator)(f::F, ncomp = nothing) where {F}
    # This is equivalent to the unnormalised trapezoidal rule,
    # assuming that the grid is evenly spaced.
    integrator.buf .= f.(integrator.xs)
    s = integrator.buf[1] + 2 * sum(integrator.buf[2 : end - 1]) + integrator.buf[end]
    BareIntegrationResult(s)
end

# Generic interface

get_grid(integrator::FixedGridIntegrator) = integrator.grid
get_grid(integrator::PreallocatedFixedGridIntegrator) = integrator.inner.grid
get_grid(integrator::IterativeFixedGridIntegrator) = integrator.grid
get_grid(integrator::MidpointIntegrator) = integrator.xs

AnyGridIntegrator = Union{FixedGridIntegrator, IterativeFixedGridIntegrator, PreallocatedFixedGridIntegrator, MidpointIntegrator}
