struct FixedGridOptimizer{ContainerT <: Union{Vector{Float64}, Vector{Vector{Float64}}}} <:
       Optimizer
    grid::ContainerT

    function FixedGridOptimizer(grid)
        new{Vector{Float64}}(grid)
    end

    function FixedGridOptimizer(grid::Vector{Vector{Float64}})
        new{Vector{Vector{Float64}}}(grid)
    end
end

function even_grid(theta_lo::Number, theta_hi::Number, quadpts)
    FixedGridOptimizer(range(theta_lo, theta_hi, quadpts))
end

function even_grid(theta_lo::AbstractVector, theta_hi::AbstractVector, quadpts_per_dim)
    prod = Iterators.product((
        range(lo, hi, length = quadpts_per_dim)
    for (lo, hi) in zip(theta_lo, theta_hi)
    )...)
    grid = reshape(collect.(prod), :)
    FixedGridOptimizer(grid)
end

function quasimontecarlo_grid(theta_lo, theta_hi, quadpts, sampler)
    grid = QuasiMonteCarlo.sample(quadpts, theta_lo, theta_hi, sampler)
    FixedGridOptimizer(grid)
end

function (optimizer::FixedGridOptimizer)(f::F) where {F}
    _, idx = findmin(x -> f(x), optimizer.grid)
    optimizer.grid[idx]
end
