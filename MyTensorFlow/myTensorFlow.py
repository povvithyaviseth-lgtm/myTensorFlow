import numpy as np

from MyTensorFlow.ActivationFunction import affine, linear, sigmoid, relu

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

def logistic_cost_function(X, y, w, b):
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z = affine(X[i], w, b)
        predict = sigmoid(z)
        cost += y[i] * np.log(predict) + (1-y[i]) * np.log(1-predict)
    return -1/m * cost

def compute_gradient(X, y, w, b, function):
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        z = affine(X[i], w, b)
        predict = function(z)
        err_i = predict - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i][j]
        dj_db += err_i


    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db