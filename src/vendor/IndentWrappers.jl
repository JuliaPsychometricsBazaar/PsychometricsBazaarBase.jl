# This module has been vendored from a pull request on
# https://github.com/tpapp/IndentWrappers.jl/pull/4
# It can be removed when equivalent functionality available elsewhere
"""
Wrapper type for indentation management for plain text printing.

The single exported function is [`indent`](@ref), see its docstring for usage.
"""
module IndentWrappers

export indent

struct IndentWrapper{T <: IO} <: Base.AbstractPipe
    parent::T
    spaces::Int
    need_indent::Ref{Bool}
    function IndentWrapper(io::T, spaces::Integer; skip_first=false) where {T <: IO}
        spaces ≥ 0 || throw(ArgumentError("negative indent not allowed"))
        new{T}(io, Int(spaces), Ref(!skip_first))
    end
end

"""
    indent(io, spaces)

Return a wrapper around `io` that prepends each `\n` written to the stream with the given
number of spaces.

It is recommended that indent is chained together in a functional manner. Blocks should
always begin with a newline and end *without one*.

# Example

```julia
julia> let io = stdout
           print(io, "toplevel")
           let io = indent(io, 4)
               print(io, '\n', "- level1")
               let io = indent(io, 4)
                   print(io, '\n', "- level 2")
               end
           end
       end
toplevel
    - level1
        - level 2
```
"""
indent(io::IO, spaces::Integer; kwargs...) = IndentWrapper(io, spaces; kwargs...)

indent(iw::IndentWrapper, spaces::Integer; kwargs...) = IndentWrapper(iw.parent, iw.spaces + spaces; kwargs...)

function Base.show(io::IO, iw::IndentWrapper)
    print(io, iw.parent, " indented by $(iw.spaces) spaces")
end

####
#### forwarded methods
####

Base.in(key_value::Pair, iw::IndentWrapper) = in(key_value, iw.parent)
Base.haskey(iw::IndentWrapper, key) = haskey(iw.parent, key)
Base.getindex(iw::IndentWrapper, key) = getindex(iw.parent, key)
Base.get(iw::IndentWrapper, key, default) = get(iw.parent, key, default)
Base.pipe_reader(iw::IndentWrapper) = iw.parent
Base.pipe_writer(iw::IndentWrapper) = iw.parent
Base.lock(iw::IndentWrapper) = lock(iw.parent)
Base.unlock(iw::IndentWrapper) = unlock(iw.parent)
Base.displaysize(iw::IndentWrapper) = displaysize(iw.parent)

####
#### capture '\n' and indent
####

_write_spaces(iw::IndentWrapper) = write(iw.parent, ' '^(iw.spaces))
function _check_need_indent(iw::IndentWrapper)
    if iw.need_indent[]
        _write_spaces(iw)
        iw.need_indent[] = false
    end
end

function Base.write(iw::IndentWrapper, chr::Char)
    _check_need_indent(iw)
    res = write(iw.parent, chr)
    if chr == '\n'
        iw.need_indent[] = true
    end
    return res
end

function Base.write(iw::IndentWrapper, str::Union{SubString{String}, String})
    _check_need_indent(iw)
    write_count = 0
    lines = split(str, '\n'; keepempty = true)
    for (i, line) in enumerate(lines)
        if i > 1
            write_count += write(iw.parent, '\n')
            if i == length(lines) && length(line) == 0
                iw.need_indent[] = true
            else
                write_count += _write_spaces(iw)
            end
        end
        write_count += write(iw.parent, line)
    end
    write_count
end

function Base.write(iw::IndentWrapper, from::AbstractVector{UInt8})
    _check_need_indent(iw)
    write_count = 0
    line_start = 1
    while true
        next_newline_idx = findnext(x -> x == 0x0A, from, line_start)
        if isnothing(next_newline_idx)
            line_end = length(from)
        else
            line_end = next_newline_idx
        end
        if line_start > 1
            if line_end < line_start
                # Empty final line
                iw.need_indent[] = true
                break
            else
                write_count += _write_spaces(iw)
            end
        end
        write_count += write(iw.parent, @view from[line_start:line_end])
        # Unterminated final line
        if isnothing(next_newline_idx)
            break
        end
        line_start = line_end + 1
    end
    write_count
end

function Base.write(to::IndentWrapper, from::StridedVector{UInt8})
    return invoke(Base.write, Tuple{IndentWrapper, AbstractVector{UInt8}}, to, from)
end

# Override dispatching to Base.write(::IO, ::GenericIOBuffer) which uses raw writing
# on the underlying fd so doesn't indent.
function Base.write(to::IndentWrapper, from::Base.GenericIOBuffer)
    invoke(Base.write, Tuple{IO, IO}, to, from)
end

end
