import numpy as np

# Calculate the pre-activation value for one neuron.
# Formula: z = w · x + b
# w: weight vector
# x: input vector
# b: scalar bias
def affine(x, w, b):
    return np.dot(w,x) + b

# Activation Function

# Relu Activation Function
# Formula: g(z) = max(0,z)
# z: Pre-activation Value
def relu(z):
    return np.maximum(0,z)

# Sigmoid Activation Function
# Formula: sigmoid(z) = 1/(1+e^(-z))
# z: Pre-activation Value
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Linear Activation Function
# Formula: Linear(z) = z = w · x + b
# z: Pre-activation Value
def linear(z):
    return z