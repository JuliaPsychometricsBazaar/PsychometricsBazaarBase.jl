module Differentiation

export vector_hessian, double_derivative

using ForwardDiff

function vector_hessian(f, x, n)
    out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
    return reshape(out, n, n, n)
end

function double_derivative(f, x)
    ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x)
end

end
