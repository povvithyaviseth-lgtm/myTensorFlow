import numpy as np
import sympy as sp

"""
Calculate the pre-activation value for one neuron.
Formula: z = w · x + b
Arg:
    w (ndarray (n,)): Weight with n Feature
    x (ndarray (n,)): Input with n Feature
    b (Scalar): Bias
Return: 
    z (scalar): w · x + b (Pre-activation Value)
"""
def affine(x, w, b):
    return np.dot(w, x) + b

"""
Relu Activation Function
Formula: g(z) = max(0,z)
Arg:
    z (scalar): Pre-activation Value
Return: 
    g (scalar): max(0,z)
"""
def relu(z):
    if isinstance(z, sp.Basic):
        return sp.Piecewise((0, z <= 0),(z, z > 0))
    else:
        return np.max(0,z)

"""
Sigmoid Activation Function
Formula: sigmoid(z) = 1/(1+e^(-z))
Arg:
    z (scalar): Pre-activation Value
Return: 
    g (scalar): 1/(1+e^(-z))
"""
def sigmoid(z):
    exp = sp.exp if isinstance(z, sp.Basic) else np.exp
    return 1 / (1 + exp(-z))

"""
Linear Activation Function
Formula: Linear(z) = z = w · x + b
Arg:
    z (scalar): Pre-activation Value
Return: 
    g (scalar): z
"""
def linear(z):
    return z

