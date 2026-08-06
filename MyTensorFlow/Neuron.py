import random

import numpy as np
from ActivationFunction import linear, affine

class Neuron:
    # Assumption: w and b has a fit value
    def __init__(self, size=0, activation=linear):
        self._w = np.random.randn(size)
        self._b = 0.3
        self._activation = activation

    def forward(self, x):
        z = affine(x, self._w, self._b)
        return self._activation(z)

    def set_w(self, w=None, b=None):
        self._w = w
        self._b = b

