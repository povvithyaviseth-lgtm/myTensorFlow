import numpy as np

from ActivationFunction import linear
from MyTensorFlow.Neuron import Neuron


class Dense:
    def __init__(self, units=0, activation=linear):
        self._neurons = [Neuron(3,activation) for _ in range(units)]

    def forward(self, x):
        return np.array([neuron.forward(x) for neuron in self._neurons])