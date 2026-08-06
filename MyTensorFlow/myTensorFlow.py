import numpy as np

from MyTensorFlow.ActivationFunction import affine, linear

"""
x_train (ndarray (m,n)): m Example with n feature
y_target (ndarray (m,)): m Target

w (ndarray (n, )): n Feature
b (scalar): Bias
"""

def fit(X, y, epochs=1000, learning_rate=0.01, activation=linear):
    m, n = X.shape

    w = np.zeros(n)
    b = 0.0

    for epoch in range(epochs):
        dj_dw, dj_db = compute_gradient(X, y, w, b, activation)
        for j in range(n):
            w[j] += learning_rate * dj_dw[j]
        b += learning_rate * dj_db

    return w,b

def compute_gradient(X, y, w, b, function):
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        f_wb_i = affine(X[i], w, b)
        predict = function(f_wb_i)
        err_i = predict - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i][j]
        dj_db += err_i


    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db