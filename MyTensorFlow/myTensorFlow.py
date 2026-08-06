import numpy as np

from MyTensorFlow.ActivationFunction import affine, linear

"""
x_train (ndarray (m,n)): m Example with n feature
y_target (ndarray (m,)): m Target

w (ndarray (n, )): n Feature
b (scalar): Bias
"""

def compute_gradient(X, y, w, b, activation=linear):
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        f_wb_i = affine(X[i], w, b)
        predict = activation(f_wb_i)
        err_i = predict - [i]
        for j in range(n):
            dj_dw[j] += err_i * X[i][j]
        dj_db += err_i

    return dj_dw, dj_db