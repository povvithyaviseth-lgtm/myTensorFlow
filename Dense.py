import numpy as np
from ActivationFunction import linear
from AutoDiff import da_dz

"""
x (ndarray (m, )): Feature m
W (ndarray (m,n)): Input m, Neuron n
b (ndarray (n,)): Neuron n
"""
class Dense:
    def __init__(self, input_size=0, units=0, activation=linear):
        self._input = None
        self._z = None
        self._a = None
        self._W = np.random.randn(input_size, units) * 0.01
        self._b = np.zeros(units)
        self._activation = activation

    def forward(self, X):
        self._input = X
        self._z = np.matmul(X, self._W) + self._b
        self._a = self._activation(self._z)
        return self._a

    """
    Calculate Backprop Using Chain Rule
    Formula: 
    dJ/dw = dJ/da * da/dz * dz/dw
    dJ/db = dJ/da * da/dz * dz/db
    
    z(j) = w(j) * a(j-1) + b(j)
    dz/dw = a(j-1) (Input)
    dz/da = w(j)
    dz/db = 1
    
    dJ/da (ndarray (n, )): Derivative of Cost Function with respect to Predicted Value
    da/dz (ndarray (n, )): Derivative of Activation Function
    dJ/dw (ndarray (m,n)): Input m, Neuron n
    dJ/db (ndarray (n, )): Neuron n
    """
    def backward(self, dJ_da):
        da_dz_func = da_dz(self._activation)
        da_dz_val = da_dz_func(self._z)

        dJ_dz = dJ_da * da_dz_val

        dJ_dw = np.outer(self._input, dJ_dz)        # dJ/dw = dJ/dz * dz/dw
        dJ_db = dJ_dz

        dJ_da_prev = np.matmul(self._W,dJ_dz)       # dJ/da = dJ/dz * dz/da
        return dJ_da_prev, dJ_dw, dJ_db

    def update(self, dJ_dw, dJ_db, learning_rate=0.01):
        self._W -= learning_rate * dJ_dw
        self._b -= learning_rate * dJ_db