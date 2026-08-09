import pandas as pd
import numpy as np

from ActivationFunction import relu, sigmoid
from Dense import Dense
from LossFunction import BinaryCrossentropy, logistic_cost_function

df = pd.read_csv("../data/student_exam_pass_dataset.csv")
X = np.array(df.drop(columns=["student_id","passed"]))
y = np.array(df["passed"])
m,n = X.shape

hidden_layer1 = Dense(input_size=n,units=n,activation=sigmoid,loss=BinaryCrossentropy)
hidden_layer2 = Dense(input_size=n,units=n,activation=sigmoid,loss=BinaryCrossentropy)
output_layer = Dense(input_size=n, units=1,activation=sigmoid, loss=BinaryCrossentropy)
# Final Prediction

learning_rate = 0.01
num_neuron = n
num_layer = 3
num_out = 1
epochs = 100
for epoch in range(epochs):

    total_cost = 0

    for i in range(len(X)):
        x = X[i]
        target = y[i]

        # Forward
        a1 = hidden_layer1.forward(x)
        a2 = hidden_layer2.forward(a1)
        a3 = output_layer.forward(a2)[0]

        cost = BinaryCrossentropy(a3, target)
        total_cost += cost

        # Output
        w3, b3 = output_layer.get_neurons()[0].get_weight()

        dJ_dz3 = a3 - target
        dJ_dw3 = a2 * dJ_dz3
        dJ_db3 = dJ_dz3

        # Layer 2
        w2 = []
        dJ_dw2 = []
        dJ_db2 = []
        dJ_dz2 = []

        dJ_da2 = dJ_dz3 * w3

        for j, neuron in enumerate(hidden_layer2.get_neurons()):
            w2_j, b2_j = neuron.get_weight()
            w2.append(w2_j)
            da2_dz2_j = a2[j] * (1 - a2[j])
            dJ_dz2_j = dJ_da2[j] * da2_dz2_j
            dJ_dw2_j = a1 * dJ_dz2_j
            dJ_db2_j = dJ_dz2_j
            dJ_dz2.append(dJ_dz2_j)
            dJ_dw2.append(dJ_dw2_j)
            dJ_db2.append(dJ_db2_j)

        # Gradient reaching Layer 1
        dJ_da1 = np.zeros_like(a1)
        for k in range(len(hidden_layer2.get_neurons())):
            dJ_da1 += dJ_dz2[k] * w2[k]

        # Layer 1
        dJ_dw1 = []
        dJ_db1 = []
        dJ_dz1 = []

        for j, neuron in enumerate(hidden_layer1.get_neurons()):
            w1_j, b1_j = neuron.get_weight()
            # assuming sigmoid
            da1_dz1_j = a1[j] * (1 - a1[j])
            dJ_dz1_j = dJ_da1[j] * da1_dz1_j
            dJ_dw1_j = x * dJ_dz1_j
            dJ_db1_j = dJ_dz1_j
            dJ_dz1.append(dJ_dz1_j)
            dJ_dw1.append(dJ_dw1_j)
            dJ_db1.append(dJ_db1_j)

        # Update Layer 1
        for j, neuron in enumerate(hidden_layer1.get_neurons()):
            w1_j, b1_j = neuron.get_weight()
            w1_j -= learning_rate * dJ_dw1[j]
            b1_j -= learning_rate * dJ_db1[j]
            neuron.set(w1_j, b1_j)

        # Update Layer 2
        for j, neuron in enumerate(hidden_layer2.get_neurons()):
            w2_j, b2_j = neuron.get_weight()
            w2_j -= learning_rate * dJ_dw2[j]
            b2_j -= learning_rate * dJ_db2[j]
            neuron.set(w2_j, b2_j)

        # Update Layer 3
        output_neuron = output_layer.get_neurons()[0]
        w3, b3 = output_neuron.get_weight()
        w3 -= learning_rate * dJ_dw3
        b3 -= learning_rate * dJ_db3
        output_neuron.set(w3, b3)


    average_cost = total_cost / len(X)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Cost: {average_cost:.6f}"
    )