from ActivationFunction import linear
from sympy import symbols, diff, lambdify

def da_dz(function=linear):
    z = symbols("z")
    a = function(z)
    da_dz = diff(a,z)
    return lambdify((z), da_dz, "numpy")