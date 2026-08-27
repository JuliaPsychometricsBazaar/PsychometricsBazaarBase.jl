"""
Mutable functor which negates a single-dimensional domain function.

The wrapped function can be swapped between optimization runs, which allows the
Optim.jl objective and state built around it to be reused.

$(TYPEDEF)
"""
mutable struct NegatedScalarFunction
    f::Any
end

(neg::NegatedScalarFunction)(θ_arr) = -neg.f(first(θ_arr))

"""
Mutable functor which negates a multi-dimensional domain function.

The wrapped function can be swapped between optimization runs, which allows the
Optim.jl objective and state built around it to be reused.

$(TYPEDEF)
"""
mutable struct NegatedVectorFunction
    f::Any
end

(neg::NegatedVectorFunction)(θ_arr) = -neg.f(θ_arr)

"""
An Optim.jl based optimizer which reuses its objective between runs.

Construct one with `preallocate(optimizer)`. Within an infinite box the Optim.jl
state is reused too, which is why the method and options are fixed when the
optimizer is preallocated. Since the objective and state are reused, an instance
must not be shared between threads or tasks.

$(TYPEDEF)
"""
struct PreallocatedOptimOptimizer{
    InnerT <: Optimizer,
    NegatedT <: Union{NegatedScalarFunction, NegatedVectorFunction},
    ObjectiveT <: Optim.AbstractObjective,
    StateT
} <: Optimizer
    inner::InnerT
    negated::NegatedT
    objective::ObjectiveT
    # The Optim.jl state, or nothing when the box has to be reapplied each run
    state::StateT
    lo::Vector{Float64}
    hi::Vector{Float64}
    x::Vector{Float64}
end

"""
$(SIGNATURES)

Preallocate the Optim.jl objective and state of a one-dimensional domain
optimizer.
"""
function preallocate(opt::OneDimOptimOptimizer)
    _preallocate_optim(opt, NegatedScalarFunction(_zero_objective), [opt.lo], [opt.hi],
        [opt.initial])
end

# The constrained formulation allocates within `optimize` itself, so there is
# nothing to hoist out of the run here.
preallocate(opt::OneDimOptimOptimizer{<:IPNewton}) = opt

"""
$(SIGNATURES)

Preallocate the Optim.jl objective and state of a multi-dimensional domain
optimizer.
"""
function preallocate(opt::MultiDimOptimOptimizer)
    _preallocate_optim(opt, NegatedVectorFunction(_zero_objective), copy(opt.lo),
        copy(opt.hi), copy(opt.initial))
end

"""
$(SIGNATURES)

Fallback for optimizers with nothing to preallocate.
"""
preallocate(opt::Optimizer) = opt

_zero_objective(θ_arr) = 0.0

function _preallocate_optim(opt, negated, lo, hi, x)
    if !_unbounded(lo, hi)
        # Fminbox rebuilds its barrier and its inner state on each run, so the
        # objective is all we can hold on to.
        if !(_box_method(opt.optim) isa Fminbox)
            return opt
        end
        objective = OnceDifferentiable(negated, x, 0.0; autodiff = :finite)
        return PreallocatedOptimOptimizer(opt, negated, objective, nothing, lo, hi, x)
    end
    objective = Optim.promote_objtype(opt.optim, x, :finite, true, negated)
    state = Optim.initial_state(opt.optim, opt.opts, objective, x)
    if !applicable(Optim.reset!, opt.optim, state, objective, x)
        # Optim.jl cannot rewind this method's state, so a fresh run is the only
        # correct option.
        return opt
    end
    PreallocatedOptimOptimizer(opt, negated, objective, state, lo, hi, x)
end

function (opt::PreallocatedOptimOptimizer)(
        f::F;
        lo = opt.inner.lo,
        hi = opt.inner.hi,
        initial = opt.inner.initial,
        optim = opt.inner.optim,
        opts = opt.inner.opts
) where {F}
    # The state belongs to the method and the box it was built for.
    if optim !== opt.inner.optim || _unbounded(lo, hi) != (opt.state !== nothing)
        return opt.inner(f; lo = lo, hi = hi, initial = initial, optim = optim,
            opts = opts)
    end
    opt.negated.f = f
    _set_bounds!(opt, lo, hi)
    _set_initial!(opt, initial)
    # Drop the values cached for the previous objective function.
    NLSolversBase.clear!(opt.objective)
    _run(opt, opt.state, optim, opts)
end

function _run(opt::PreallocatedOptimOptimizer, ::Nothing, optim, opts)
    _minimizer(opt,
        optimize(opt.objective, opt.lo, opt.hi, opt.x, _box_method(optim), opts))
end

function _run(opt::PreallocatedOptimOptimizer, state, optim, opts)
    if hasproperty(state, :x)
        # Optim.reset! warm starts from the state's own iterate, which is the
        # minimizer of the previous run.
        copyto!(state.x, opt.x)
    end
    Optim.reset!(optim, state, opt.objective, opt.x)
    _minimizer(opt, optimize(opt.objective, opt.x, optim, opts, state))
end

function _set_initial!(opt::PreallocatedOptimOptimizer{<:OneDimOptimOptimizer}, initial)
    opt.x[1] = initial
end

function _set_initial!(opt::PreallocatedOptimOptimizer{<:MultiDimOptimOptimizer}, initial)
    copyto!(opt.x, initial)
end

function _set_bounds!(opt::PreallocatedOptimOptimizer{<:OneDimOptimOptimizer}, lo, hi)
    opt.lo[1] = lo
    opt.hi[1] = hi
end

function _set_bounds!(opt::PreallocatedOptimOptimizer{<:MultiDimOptimOptimizer}, lo, hi)
    copyto!(opt.lo, lo)
    copyto!(opt.hi, hi)
end

_minimizer(::PreallocatedOptimOptimizer{<:OneDimOptimOptimizer}, res) = Optim.minimizer(res)[1]
_minimizer(::PreallocatedOptimOptimizer{<:MultiDimOptimOptimizer}, res) = Optim.minimizer(res)
