import numpy as np
import sympy as sp
from ActivationFunction import sigmoid, affine

def logistic_cost_function(X, y, w, b):
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z = affine(X[i], w, b)
        predict = sigmoid(z)
        cost += BinaryCrossentropy(predict, y[i])
    return 1/m * cost

def MSE(a, y):
    return 1 / 2 * (a - y) ** 2

def log(x):
    return sp.log(x) if isinstance(x, sp.Basic) else np.log(x)

def BinaryCrossentropy(a, y):
    return -(y * log(a) + (1 - y) * log(1 - a))

def dJ_da(loss=MSE):
    a, y = sp.symbols("a, y")
    J = loss(a, y)
    dJ_da = sp.diff(J, a)
    return sp.lambdify((a, y), dJ_da, "numpy")