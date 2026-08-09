from LossFunction import BinaryCrossentropy, dJ_da

"""
layers (List of Dense)
"""
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

    def fit(self,X, y, epochs=100):
        dJ_da_func = dJ_da(loss=BinaryCrossentropy)
        for epoch in range(epochs):
            total_cost = 0
            for i in range(len(X)):
                x = X[i]
                target = y[i]

                a = self.forward(x)
                total_cost += BinaryCrossentropy(a,target) # Change this Later

                dJ_da_i = dJ_da_func(a, target)
                gradients = []
                for layer in reversed(self._layers):
                    dJ_da_i, dJ_dw, dJ_db = layer.backward(dJ_da_i)
                    gradients.append((layer, dJ_dw, dJ_db))

                for layer, dJ_dw, dJ_db in gradients:
                    layer.update(dJ_dw, dJ_db)

            average_cost = total_cost / len(X)
            print(f"Epoch {epoch + 1}: Cost = {average_cost}")