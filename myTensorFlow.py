import numpy as np


from ActivationFunction import affine, linear, sigmoid, relu
from LossFunction import dJ_da, MSE, BinaryCrossentropy
from AutoDiff import da_dz, linear_sym, sigmoid_sym, relu_sym

"""
x_train (ndarray (m,n)): m Example with n feature
y_target (ndarray (m,)): m Target

w (ndarray (n, )): n Feature
b (scalar): Bias
"""

activation_sym = {linear: linear_sym,
                  sigmoid: sigmoid_sym,
                  relu: relu_sym}

def fit(neuron, X, y, epochs=1000, learning_rate=0.01, loss=MSE):
    m, n = X.shape
    w, b = neuron.get_weight()
    activation = neuron.get_activation()
    for epoch in range(epochs):
        dj_dw, dj_db = compute_gradient(X, y, w, b, activation, loss, activation_sym[activation])
        for j in range(n):
            w[j] -= learning_rate * dj_dw[j]
        b -= learning_rate * dj_db
    neuron.set(w,b)

def compute_gradient(X, y, w, b, function, loss, sym):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    dj_da_diff = dJ_da(loss)
    da_dz_diff = da_dz(sym)
    for i in range(m):

        z = affine(X[i], w, b)
        predict = function(z)

        dj_da_val = dj_da_diff(predict, y[i])
        da_dz_val = da_dz_diff(z)
        dj_dz_val = dj_da_val * da_dz_val

        for j in range(n):
            dz_dw_val = X[i][j]
            dj_dw_val = dj_dz_val * dz_dw_val
            dj_dw[j] += dj_dw_val

        dz_db_val = 1
        dj_db_val = dj_dz_val * dz_db_val
        dj_db += dj_db_val

    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db