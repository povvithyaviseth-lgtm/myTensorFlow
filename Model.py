
class Model:
    def __init__(self, layers):
        self._layers = layers

    """
    return a List of Activation Result
    """
    def forward(self, x):
        activations = [x]
        a = x
        for layer in self._layers:
            a = layer.forward(a)
            activations.append(a)
        return activations