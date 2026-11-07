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
