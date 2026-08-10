import sympy as sp
from ActivationFunction import linear
from LossFunction import MSE

def da_dz(function=linear):
    z = sp.symbols("z")
    a = function(z)
    da_dz = sp.diff(a,z)
    return sp.lambdify((z), da_dz, "numpy")

def dJ_da(loss=MSE):
    a, y = sp.symbols("a, y")
    J = loss(a, y)
    dJ_da = sp.diff(J, a)
    return sp.lambdify((a, y), dJ_da, "numpy")