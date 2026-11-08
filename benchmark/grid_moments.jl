# Run with julia --project=. benchmark/grid_moments.jl (one thread).
using PsychometricsBazaarBase.Integrators
using Printf

function scan(x, w, repetitions)
    total = 0.0
    for _ in 1:repetitions
        total += weighted_grid_moments(x, w).variance
    end
    total
end

function benchmark_moments()
    println("Julia ", VERSION, ", threads=", Threads.nthreads())
    println("points, setup bytes, setup ns, reduction bytes, reduction ns")
    for n in (33, 65, 129)
        setup() = (collect(range(-4., 4.; length=n)), ones(n))
        setup() # compile setup separately
        prepared = @timed setup()
        x, w = prepared.value
        scan(x, w, 1) # warm reduction
        bytes = @allocated scan(x, w, 10000)
        elapsed = minimum(@elapsed(scan(x, w, 10000)) for _ in 1:5)
        @printf("%d, %d, %.0f, %.0f, %.1f\n", n, prepared.bytes,
            prepared.time * 1e9, bytes / 10000, elapsed * 1e9 / 10000)
    end
end
benchmark_moments()

function transformed_scan(factors, x, w, buffers, repetitions, fused)
    total = 0.0
    for _ in 1:repetitions
        if fused
            summaries = weighted_grid_moments!(factors, x, w, buffers)
            total += summaries[1].variance + summaries[2].variance
        else
            for i in eachindex(x, w)
                values = factors(x[i])
                buffers[1][i] = w[i] * values[1]
                buffers[2][i] = w[i] * values[2]
            end
            total += weighted_grid_moments(x, buffers[1]).variance +
                weighted_grid_moments(x, buffers[2]).variance
        end
    end
    total
end

function benchmark_transformed_moments()
    println("transform, points, fused, setup bytes, setup ns, reduction bytes, reduction ns")
    for (name, factors) in (("constant", x -> (1.0, 2.0)),
            ("exp", x -> (exp(x), exp(-x)))), n in (33, 65, 129)
        setup() = (collect(range(-4., 4.; length=n)), ones(n), (zeros(n), zeros(n)))
        setup()
        prepared = @timed setup()
        x, w, buffers = prepared.value
        for fused in (false, true)
            transformed_scan(factors, x, w, buffers, 1, fused)
            bytes = @allocated transformed_scan(factors, x, w, buffers, 10000, fused)
            elapsed = minimum(@elapsed(transformed_scan(factors, x, w, buffers, 10000, fused)) for _ in 1:5)
            @printf("%s, %d, %s, %d, %.0f, %.0f, %.1f\n", name, n, fused,
                prepared.bytes, prepared.time * 1e9, bytes / 10000, elapsed * 1e9 / 10000)
        end
    end
end
benchmark_transformed_moments()
