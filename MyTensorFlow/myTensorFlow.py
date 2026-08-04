import numpy as np

# Calculate the pre-activation value for one neuron.
# Formula: z = w · x + b
# w: weight vector
# x: input vector
# b: scalar bias
# z: scalar pre-activation value
def affine(x, w, b):
    return np.dot(w,x) + b

