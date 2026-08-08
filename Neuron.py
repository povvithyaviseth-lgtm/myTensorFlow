import numpy as np
from ActivationFunction import linear, affine
from MyTensorFlow.LossFunction import MSE

"""
x (ndarray (n, )): Input Value with n feature

_w (ndarray (n, )): Weight with n feature
_b (scalar): Bias
"""

class Neuron:
    def __init__(self, num_feat=0, activation=linear, loss=MSE):
        self._w = np.zeros(num_feat)
        self._b = 0.0
        self._activation = activation
        self._loss = loss

    def forward(self, x):
        pre_activation = affine(x, self._w, self._b)
        return self._activation(pre_activation)

    def set(self, w=None, b=None):
        self._w = w
        self._b = b

    def get_weight(self):
        return self._w, self._b

    def get_activation(self):
        return self._activation

    def get_loss(self):
        return self._loss

    def predict(self, x):
        return self.forward(x)