import numpy as np

from LossFunction import MSE
from Neuron import Neuron
from ActivationFunction import linear

class Dense:
    def __init__(self, input_size=0, units=0, activation=linear, loss=MSE):
        self._neurons = [Neuron(input_size=input_size,activation=activation,loss=loss)
                         for _ in range(units)]

    def forward(self, X):
        output = [neuron.forward(X) for neuron in self._neurons]
        return np.array(output)

    def get_neurons(self):
        return self._neurons