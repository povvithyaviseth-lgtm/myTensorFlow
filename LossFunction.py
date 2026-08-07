from ActivationFunction import relu, sigmoid, affine, linear
from sympy import symbols, diff, exp, log, lambdify


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

def BinaryCrossentropy(a, y):
    return -(y * log(a) + (1 - y) * log(1 - a))

def dJ_da(loss=MSE):
    a, y = symbols("a, y")
    J = loss(a, y)
    dJ_da = diff(J, a)
    return lambdify((a, y), dJ_da, "numpy")