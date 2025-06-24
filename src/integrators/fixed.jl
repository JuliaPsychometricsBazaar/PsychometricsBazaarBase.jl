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

struct PreallocatedFixedGridIntegrator{ContainerT <:
                                       Union{Vector{Float64}, Matrix{Float64}},
                                       FixedGridIntegratorT <: FixedGridIntegrator} <:
       Integrator
    inner::FixedGridIntegratorT
    buf::ContainerT

    function PreallocatedFixedGridIntegrator(inner::FixedGridIntegrator{Vector{Float64}})
        quadpts = length(inner.grid)
        buf = Vector{Float64}(undef, quadpts)
        new{Vector{Float64}, FixedGridIntegrator{Vector{Float64}}}(inner, buf)
    end

    function PreallocatedFixedGridIntegrator(inner::FixedGridIntegrator{Vector{Vector{Float64}}})
        quadpts = length(inner.grid)
        # XXX: In general this is wrong. The output dimension could be anything.
        # TODO: Instead it be that we have a maximum output dimension specified at construction time
        dim = length(inner.grid[1])
        buf = Matrix{Float64}(undef, quadpts, dim)
        new{Matrix{Float64}, FixedGridIntegrator{Vector{Vector{Float64}}}}(inner, buf)
    end
end

function (integrator::PreallocatedFixedGridIntegrator{Vector{Float64}})(
    f::F,
    ncomp::Int = 0
) where {F}
    if ncomp == 0 || ncomp == 1
        integrator.buf .= f.(integrator.inner.grid)
        BareIntegrationResult(sum(integrator.buf))
    else
        error("ncomp must be 0 or 1 for FixedGridIntegrator with Vector{Float64} buffer")
    end
end

function (integrator::PreallocatedFixedGridIntegrator{Matrix{Float64}})(
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

struct IterativeFixedGridIntegrator{ContainerT <:
                                    Union{Vector{Float64}, Vector{Vector{Float64}}}} <:
       Integrator
    grid::ContainerT
end

function (integrator::IterativeFixedGridIntegrator)(f::F, ncomp = nothing) where {F}
    s = sum(f, integrator.grid)
    BareIntegrationResult(s)
end

function show(io::IO, ::MIME"text/plain", integrator::Union{FixedGridIntegrator, IterativeFixedGridIntegrator})
    if integrator.grid isa AbstractRange
        println(io, "Fixed step grid integrator:")
    else
        println(io, "Array-based grid integrator:")
    end
    println(io, "  Number of integration points: ", length(integrator.grid))
    println(io, "  Dimensions: ", length(integrator.grid[1]))
    if integrator.grid isa AbstractRange
        println(io, "  Start: ", first(integrator.grid))
        println(io, "  End: ", last(integrator.grid))
        println(io, "  Step size: ", step(integrator.grid))
    else
        println(io, "  Grid:")
        buf = IOBuffer()
        show(IOContext(buf, :limit => true, :displaysize => (10, 10)), MIME("text/plain"), integrator.grid)
        seekstart(buf)
        for line in eachline(buf)
            println(io, "    ", line)
        end
    end
end

show(io::IO, ::MIME"text/plain", integrator::PreallocatedFixedGridIntegrator) = show(io, MIME("text/plain"), integrator)

struct MidpointIntegrator <: Integrator
    xs::Vector{Float64}
    buf::Vector{Float64}

    function MidpointIntegrator(xs)
        buf = Vector{Float64}(undef, length(xs))
        new(xs, buf)
    end
end

function (integrator::MidpointIntegrator)(f::F, ncomp = nothing) where {F}
    # This is equivalent to the unnormalised trapezoidal rule,
    # assuming that the grid is evenly spaced.
    integrator.buf .= f.(integrator.xs)
    s = integrator.buf[1] + 2 * sum(integrator.buf[2 : end - 1]) + integrator.buf[end]
    BareIntegrationResult(s)
end
