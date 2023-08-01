"""
This module contains utilities to implement highly configurible library code
where configuration is performed through structs, and smart defaults allow
sloppy or flat specification of otherwise deeply nested configuration structs.
"""
module ConfigTools

export @requiresome, @returnsome
export find1, find1_instance, find1_type, find1_type_sloppy

using MacroTools
using DocStringExtensions

"""
$(SIGNATURES)

This macro is passed an assignment like so

    @requiresome foo = bar()

If `bar()` returns `nothing`, then the macro causes the current function to
return `nothing`. Otherwise, execution continues.
"""
macro requiresome(assign)
    @capture(assign, name_ = expr_) || error("@requiresome must be passed an assignment")
    quote
        $(esc(assign))
        if $(esc(name)) === nothing
            return nothing
        end
    end
end

"""
$(SIGNATURES)

This macro is passed an expression like so

    @returnsome foo()

If `foo()` return any value apart from `nothing`, the macro causes the current
function to return that value. Otherwise, execution continues.
"""
macro returnsome(expr)
    quote
        val = $(esc(expr))
        if val !== nothing
            return val
        end
    end
end

"""
$(SIGNATURES)

This macro is passed an expression and a function like so

    @returnsome foo() do x
        bar(x)
    end

If `foo()` return any value apart from `nothing`, the macro executes the
function and returns the value as long as it is not `nothing`. In all other
cases, execution continues.
"""
macro returnsome(expr, func)
    quote
        val = $(esc(expr))
        if val !== nothing
            res = ($(esc(func)))(val)
            if res !== nothing
                return res
            end
        end
    end
end

"""
$(SIGNATURES)

Given an iterable `iter` and a predicate `pred`, this function returns a match
or else `nothing` if no match. In case there are multiple matches, an error is
thrown.
"""
function find1(pred::F, iter, fail_msg) where {F}
    res = nothing
    cnt = 0
    for bit in iter
        if pred(bit)
            res = bit
            cnt += 1
        end
    end
    if cnt > 1
        error(fail_msg)
    end
    return res
end

"""
$(SIGNATURES)

Returns exactly one instance in `iter` of type `type` or else `nothing``. In
case there are multiple matches, an error is thrown.
"""
function find1_instance(type, iter)
    find1(
        x -> isa(x, type),
        iter,
        "Expected exactly one instance of " * repr(type)
    )
end

"""
$(SIGNATURES)

Returns exactly one type in `iter` of which is a subtype of `type` or else
`nothing``.  In case there are multiple matches, an error is thrown.
"""
function find1_type(type, iter)
    return find1(
        x -> (((x isa DataType) || (x isa UnionAll)) && x <: type),
        iter,
        "Expected exactly one type " * repr(type)
    )
end

"""
$(SIGNATURES)

Returns exactly one type in `iter` of which is either a subtype of `type` or an
instance of `type` or else `nothing``.  In case there are multiple matches, an
error is thrown.
"""
function find1_type_sloppy(type, iter)
    @returnsome find1_type(type, iter)
    @returnsome find1_instance(type, iter) inst -> typeof(inst)
end

end