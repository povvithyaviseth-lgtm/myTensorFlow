from LossFunction import SE
from AutoDiff import dJ_da

"""
Sequential Model
Args:
    layers (list of Dense): List of Dense Layers in Forward Order

Attributes:
    _layers        (list of Dense): Layers Used in the Model
    _loss              (function): Loss Function Used During Training
    _learning_rate      (scalar): Step Size Used During Gradient Descent
"""
class Sequential:
    def __init__(self, layers):
        self._layers = layers
        self._loss = SE
        self._learning_rate = 0.1

    """
    Calculate Forward Propagation Through All Layers

    Formula:
        a[0] = x
        a[1] = Layer1(a[0])
        a[2] = Layer2(a[1])
        ...
        a[L] = LayerL(a[L-1])

    Args:
        x (ndarray (m, )):
            Input Sample to the Model

    Return:
        a (ndarray (n, )):
            Final Activation / Prediction from Output Layer
    """
    def forward(self, x):
        a = x
        for layer in self._layers:
            a = layer.forward(a)
        return a

    """
    Configure the Model for Training

    Args:
        learning_rate (scalar):
            Step Size Used During Gradient Descent

        loss (function):
            Loss Function Used to Calculate Cost

    Return:
        None
    """
    def compile(self, learning_rate=0.1, loss=SE):
        self._loss = loss
        self._learning_rate = learning_rate


    """
    Train the Model Using Forward Propagation,
    Backpropagation, and Gradient Descent

    Formula:
        Forward Propagation:
            x -> Layer 1 -> Layer 2 -> ... -> Prediction

        Cost:
            J = 1/m * summation(Loss(a[i], y[i]))

        Backpropagation:
            dJ/da -> Last Layer -> ... -> First Layer

        Gradient Descent:
            W = W - learning_rate * dJ/dW
            b = b - learning_rate * dJ/db

    Args:
        X      (ndarray (m,n)): Training Data, m = Number of Samples, n = Number of Input Features
        y      (ndarray (m, )): Actual Target Values
        epochs        (scalar): Number of Times the Model Trains Through the Entire Dataset
    Return:
        None
    """
    def fit(self, X, y, epochs=100):
        dJ_da_func = dJ_da(loss=self._loss)                         # Create Derivative Function dJ/da
        for epoch in range(epochs):
            total_cost = 0
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                a = self.forward(x)                                 # Forward Propagation
                total_cost += self._loss(a, target)                 # Calculate Loss for Current Sample

                dJ_da_i = dJ_da_func(a, target)                     # Calculate Initial Gradient from Loss Function
                gradients = []
                for layer in reversed(self._layers):                # Move Backward Through Every Layer
                    dJ_da_i, dJ_dw, dJ_db = layer.backward(dJ_da_i)
                    gradients.append((layer, dJ_dw, dJ_db))          # Save Gradient for Current Layer

                for layer, dJ_dw, dJ_db in gradients:                # Update Weight and Bias for Every Layer
                    layer.update(dJ_dw, dJ_db,self._learning_rate)

            average_cost = total_cost / len(X)
            print(f"Epoch {epoch + 1}: Cost = {average_cost}" )
