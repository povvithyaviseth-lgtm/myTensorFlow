from MyTensorFlow.ActivationFunction import *


class Neuron:
    def __init__(self, size=0, units=0 ,activation=linear):
        self._w = np.zeros(size,units)
        self._b = 0.0
        self._activation = activation

    def forward(self, x):
        z = affine(x, self._w, self._b)
        return self._activation(z)