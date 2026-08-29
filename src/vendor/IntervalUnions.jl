# This module has been vendored from
# https://github.com/JuliaAPlavin/IntervalUnions.jl
# It can be removed if IntervalUnions.jl becomes a suitable direct dependency.
module IntervalUnions

using IntervalSets
using Accessors
using DataPipes

export IntervalUnion, complement, ∪ᵢ, setdiffᵢ

struct IntervalUnion{T} <: IntervalSets.Domain{T}
    ints::T
end

intervals(x::Interval) = (x,)
@accessor intervals(x::IntervalUnion) = x.ints

_is_intervals_equal(a, b) =
    if length(a) == length(b)
        all(splat(==), zip(a, b))
    elseif length(a) == 0 || length(b) == 0
        all(isempty, a) && all(isempty, b)
    else
        false
    end

Base.:(==)(a::IntervalUnion, b::IntervalUnion) = _is_intervals_equal(intervals(a), intervals(b))
Base.:(==)(a::IntervalUnion, b::Interval) = _is_intervals_equal(intervals(a), intervals(b))
Base.:(==)(a::Interval, b::IntervalUnion) = _is_intervals_equal(intervals(a), intervals(b))

IntervalUnion(x::Union{Interval, IntervalUnion}) = convert(IntervalUnion, x)
Base.convert(::Type{IntervalUnion}, x::IntervalUnion) = x
Base.convert(::Type{IntervalUnion}, x::Interval) = IntervalUnion((x,))

Base.in(x, iu::IntervalUnion) = any(in(x, i) for i in intervals(iu))
Base.isempty(iu::IntervalUnion) = all(isempty, intervals(iu))
_dropempty(iu::IntervalUnion) = IntervalUnion(filter(!isempty, intervals(iu)))

_opposite_closedness(x::Symbol) = x == :closed ? :open : x == :open ? :closed : error()
_setdiff(a::Interval{La, Ra}, b::Interval{Lb, Rb}) where {La, Ra, Lb, Rb} = intersect(a, complement(b))

complement(i::Interval{L,R,T}) where {T,L,R} = IntervalUnion((
    Interval{:open, _opposite_closedness(L)}(typemin(float(T)), leftendpoint(i)),
    Interval{_opposite_closedness(R), :open}(rightendpoint(i), typemax(float(T))),
))

# de Morgan: complement of a union is the intersection of complements
complement(iu::IntervalUnion) = @p intervals(iu) |> map(complement) |> reduce(∩)

Base.setdiff(a::Interval, b::IntervalUnion) =
    isempty(intervals(b)) ?
        IntervalUnion((a,)) :
        @p intervals(b) |> map(_setdiff(a, _)) |> reduce(∩) |> _dropempty
Base.setdiff(a::IntervalUnion, b::Union{Interval, IntervalUnion}) =
    isempty(intervals(a)) ?
        a :
        @p intervals(a) |> map(setdiff(_, IntervalUnion(b))) |> reduce(∪)
Base.:∪(a::Interval, b::IntervalUnion) = IntervalUnion((intervals(a)..., intervals(b)...))
Base.:∪(a::IntervalUnion, b::Union{Interval, IntervalUnion}) = IntervalUnion((intervals(a)..., intervals(b)...))
Base.:∩(a::Interval, b::IntervalUnion) = @modify(i -> a ∩ i, intervals(b) |> Elements()) |> _dropempty
Base.:∩(a::IntervalUnion, b::Interval) = @modify(i -> b ∩ i, intervals(a) |> Elements()) |> _dropempty
Base.:∩(a::IntervalUnion, b::IntervalUnion) =
    isempty(intervals(a)) ?
        a :
        @p intervals(a) |> map(_ ∩ b) |> reduce(∪)

IntervalSets.infimum(iu::IntervalUnion) = minimum(infimum, intervals(iu))
IntervalSets.supremum(iu::IntervalUnion) = maximum(supremum, intervals(iu))
Base.minimum(iu::IntervalUnion) = minimum(minimum, intervals(iu))
Base.maximum(iu::IntervalUnion) = maximum(maximum, intervals(iu))
Base.extrema(iu::IntervalUnion) = (minimum(iu), maximum(iu))

function Base.show(io::IO, iu::IntervalUnion)
    isempty(intervals(iu)) && return print(io, "∅")
    for (i, interval) in enumerate(intervals(iu))
        i > 1 && print(io, " ∪ ")
        print(io, interval)
    end
end

∪ᵢ(ints...) = ∪(convert.(IntervalUnion, ints)...)
setdiffᵢ(ints...) = setdiff(convert.(IntervalUnion, ints)...)

end
