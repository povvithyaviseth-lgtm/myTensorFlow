from sympy import symbols, exp, diff, lambdify, Piecewise

def relu_sym(z):
    return Piecewise(
        (0, z <= 0),
        (z, z > 0)
    )

def sigmoid_sym(z):
    return 1/(1+exp(-z))

def linear_sym(z):
    return z

def da_dz(function=linear_sym):
    z = symbols("z")
    a = function(z)
    da_dz = diff(a,z)
    return lambdify((z), da_dz, "numpy")