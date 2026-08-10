import numpy as np
import sympy as sp
from ActivationFunction import sigmoid, affine

"""
Calculate the Logistic Cost Function
Formula: Total Cost = 1/m summation(i=0 to i=m) of Loss(a[i],y[i])
Args:
    X (ndarray (m,n)): Sample m, Feature n
    y (ndarray (m, )): Sample m
    w        (scalar): Weight
    b        (scalar): Bias
Return:
    total_cost(scalar): Total Cost
"""
def logistic_cost_function(X, y, w, b):
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z = affine(X[i], w, b)
        predict = sigmoid(z)
        cost += BinaryCrossentropy(predict, y[i])
    return 1/m * cost

"""
Calculate Square Error (SE)
Formula: SE = (a - y) ^ 2
Args:
    a  (scalar): Prediction Value
    y  (scalar): Actual Value
Return:
    SE (scalar): Square Error
"""
def SE(a, y):
    return (a - y) ** 2

"""
Helper Function to Define Log
Arg:
    x       (scalar): Real Number
Return:
    log(x)  (scalar): log of x
"""
def log(x):
    return sp.log(x) if isinstance(x, sp.Basic) else np.log(x)

"""
Calculate Binary Cross-Entropy
Formula: Binary Cross-Entropy = -y * log(a) - (1-y) * log(1-a)
Args:
    a                 (scalar): Prediction Value
    y                 (scalar): Actual Value
Return:
    BinaryCrossentory (scalar): Binary Cross-Entropy
"""
def BinaryCrossentropy(a, y):
    return -(y * log(a) + (1 - y) * log(1 - a))