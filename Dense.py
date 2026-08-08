import numpy as np

from LossFunction import MSE
from Neuron import Neuron
from ActivationFunction import linear

class Dense:
    def __init__(self, num_feature=0, units=0, activation=linear, loss=MSE):
        self._neurons = [Neuron(num_feat=num_feature,activation=activation,loss=loss)
                         for _ in range(units)]

    def forward(self, X):
        output = [neuron.forward(X) for neuron in self._neurons]
        return np.array(output)