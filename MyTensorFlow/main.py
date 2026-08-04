from Neuron import *

x = np.array([1.0, 2.0])

neuron = Neuron(x.size, activation=relu)
print(f"output: {neuron.forward(x)}")