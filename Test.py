import pandas as pd
import numpy as np

from ActivationFunction import relu, sigmoid
from LossFunction import logistic_cost_function, BinaryCrossentropy
from Neuron import Neuron
from myTensorFlow import fit

num_batch = 5

df = pd.read_csv("../data/student_exam_pass_dataset.csv")
X = np.array(df.drop(columns=["student_id","passed"]))
y = np.array(df["passed"])

x_batch = np.split(X, num_batch)
y_batch = np.split(y, num_batch)

n = X.shape[1]
myNeuron = Neuron(num_feat=n, activation=sigmoid, loss=BinaryCrossentropy)

for i in range(len(x_batch) - 1):
    fit(myNeuron, x_batch[i], y_batch[i], learning_rate=1e-4)
    w, b = myNeuron.get_weight()
    print(f"{i} = w:{w}, b:{b}, cost: {logistic_cost_function(x_batch[i], y_batch[i],w,b)}")

err = 0
threshold = 0.5
for i in range(len(x_batch[-1])):
    target = y_batch[-1][i]
    predict = 1 if myNeuron.predict(x_batch[-1][i]) > threshold else 0
    err += abs(predict - target)

print(f"Error: {err}, Accuracy: {1-err/len(x_batch[-1])}")