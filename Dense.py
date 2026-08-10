import numpy as np
from ActivationFunction import linear
from AutoDiff import da_dz

""" 
Dense Layer 
Args: 
    input_size    (scalar): Number of Inputs to the Layer 
    units         (scalar): Number of Neurons in the Layer 
    activation  (function): Activation Function Used by the Layer 
Attributes: 
    _input (ndarray (m, )): Input Value to Current Layer 
    _z (ndarray     (n, )): Pre-Activation Value 
    _a (ndarray     (n, )): Activation Value / Output of Current Layer 
    _W (ndarray     (m,n)): Weight Matrix, Input m x Neuron n 
    _b (ndarray     (n, )): Bias for Each Neuron 
"""
class Dense:
    def __init__(self, input_size=0, units=0, activation=linear):
        self._input = None
        self._z = None
        self._a = None
        self._W = np.random.randn(input_size, units) * 0.01
        self._b = np.zeros(units)
        self._activation = activation

    """ 
    Calculate Forward Propagation of Dense Layer 
    Formula: z = XW + b a = g(z) 
    Arg: 
        X (ndarray (m, )): Input to Current Layer 
    Return: 
        a (ndarray (n, )): Activation Value / Output of Current Layer 
    """
    def forward(self, X):
        self._input = X
        self._z = np.matmul(X, self._W) + self._b
        self._a = self._activation(self._z)
        return self._a

    """ 
    Calculate Backpropagation Using Chain Rule 
    Formula:    dJ/dw = dJ/da * da/dz * dz/dw 
                dJ/db = dJ/da * da/dz * dz/db 
                
                z(j) = a(j-1)W(j) + b(j) 
                dz/dw = a(j-1) dz/da = W(j) 
                dz/db = 1 dJ/dz = dJ/da * da/dz           
    Args: 
            dJ_da       (ndarray (n, )): Derivative of Cost Function with Respect to Activation of Current Layer 
    Return: 
            dJ_da_prev  (ndarray (m, )): Derivative of Cost Function with Respect to Activation of Previous Layer 
            dJ_dw (ndarray       (m,n)): Derivative of Cost Function with Respect to Every Weight in Current Layer 
            dJ_db (ndarray       (n, )): Derivative of Cost Function with Respect to Every Bias in Current Layer    
    Shape: 
        input       (m, ) 
        W           (m,n) 
        b           (n, ) 
        z           (n, ) 
        a           (n, ) 
        dJ/da       (n, ) 
        da/dz       (n, ) 
        dJ/dz       (n, ) 
        dJ/dw       (m,n) 
        dJ/db       (n, ) 
        dJ/da_prev  (m, ) 
    """
    def backward(self, dJ_da):
        da_dz_func = da_dz(self._activation)
        da_dz_val = da_dz_func(self._z)

        dJ_dz = dJ_da * da_dz_val

        dJ_dw = np.outer(self._input, dJ_dz)        # dJ/dw = dJ/dz * dz/dw
        dJ_db = dJ_dz

        dJ_da_prev = np.matmul(self._W,dJ_dz)       # dJ/da = dJ/dz * dz/da
        return dJ_da_prev, dJ_dw, dJ_db

    """ 
    Update Weight and Bias Using Gradient Descent 
    Formula:    W = W - learning_rate * dJ/dW 
                b = b - learning_rate * dJ/db 
    Args: 
        dJ_dw (ndarray  (m,n)): Derivative of Cost Function with Respect to Weight 
        dJ_db (ndarray  (n, )): Derivative of Cost Function with Respect to Bias 
        learning_rate (scalar): Step Size Used During Gradient Descent 
    Return: 
        None 
    """
    def update(self, dJ_dw, dJ_db, learning_rate=0.01):
        self._W -= learning_rate * dJ_dw
        self._b -= learning_rate * dJ_db