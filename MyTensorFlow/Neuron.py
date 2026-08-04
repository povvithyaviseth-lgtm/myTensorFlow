import numpy as np
from MyTensorFlow.ActivationFunction import *


class Neuron:
    def __init__(self, size=0 ,activation=linear):
        self.w = np.zeros(size)
        self.b = 0.0
        self.activation = activation

    def forward(self, x):
        z = affine(x, self.w, self.b)
        return self.activation(z)