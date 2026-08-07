import numpy as np
from sympy.integrals.risch import derivation

from ActivationFunction import affine, linear, sigmoid, relu
from LossFunction import dJ_da, MSE, BinaryCrossentropy

"""
x_train (ndarray (m,n)): m Example with n feature
y_target (ndarray (m,)): m Target

w (ndarray (n, )): n Feature
b (scalar): Bias
"""

def fit(X, y, epochs=1000, learning_rate=0.01, activation=linear, loss=MSE):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    for epoch in range(epochs):
        dj_dw, dj_db = compute_gradient(X, y, w, b, activation, loss)
        for j in range(n):
            w[j] -= learning_rate * dj_dw[j]
        b -= learning_rate * dj_db
    return w,b

def compute_gradient(X, y, w, b, function, loss):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    dj_da = dJ_da(loss)
    for i in range(m):
        z = affine(X[i], w, b)
        predict = function(z)
        diff = dj_da(predict, y[i])
        for j in range(n):
            dj_dw[j] += diff * X[i][j]
        dj_db += diff
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db