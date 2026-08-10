import sympy as sp
from ActivationFunction import linear
from LossFunction import MSE

"""
Calculate the Derivative of Activation Function with Respect to Pre-Activation Function (da/dz)
Arg:
    function (g(z)): Activation Function
Return:
    function (g'(z)): Derivative of Activation Function
"""
def da_dz(function=linear):
    z = sp.symbols("z")
    a = function(z)
    da_dz = sp.diff(a,z)
    return sp.lambdify((z), da_dz, "numpy")

"""
Calculate the Derivative of Loss Function with Respect to Activation Function (dJ/da)
Arg:
    function (J(a)): Loss Function
Return:
    function (J'(a)): Derivative of Loss Function
"""
def dJ_da(loss=MSE):
    a, y = sp.symbols("a, y")
    J = loss(a, y)
    dJ_da = sp.diff(J, a)
    return sp.lambdify((a, y), dJ_da, "numpy")