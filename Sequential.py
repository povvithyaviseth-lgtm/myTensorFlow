from LossFunction import dJ_da, MSE

"""
layers (List of Dense)
"""
class Sequential:
    def __init__(self, layers):
        self._layers = layers
        self._loss = MSE

    """
    return a List of Activation Result
    """
    def forward(self, x):
        a = x
        for layer in self._layers:
            a = layer.forward(a)
        return a

    def compile(self,loss=MSE):
        self._loss=loss

    def fit(self,X, y, epochs=100):
        dJ_da_func = dJ_da(loss=self._loss)
        for epoch in range(epochs):
            total_cost = 0
            for i in range(len(X)):
                x = X[i]
                target = y[i]

                a = self.forward(x)
                total_cost += self._loss(a,target) # Change this Later

                dJ_da_i = dJ_da_func(a, target)
                gradients = []
                for layer in reversed(self._layers):
                    dJ_da_i, dJ_dw, dJ_db = layer.backward(dJ_da_i)
                    gradients.append((layer, dJ_dw, dJ_db))

                for layer, dJ_dw, dJ_db in gradients:
                    layer.update(dJ_dw, dJ_db)

            average_cost = total_cost / len(X)
            print(f"Epoch {epoch + 1}: Cost = {average_cost}")