import numpy as np

from ActivationFunction import linear
from MyTensorFlow.Neuron import Neuron


class Dense:
    def __init__(self, units=0, activation=linear):
        self._units = units
        self._activation = activation

    def forward(self, x):
        neurons = [Neuron(x.size,self._activation) for _ in range(self._units)]
        return np.array([neuron.forward(x) for neuron in neurons])