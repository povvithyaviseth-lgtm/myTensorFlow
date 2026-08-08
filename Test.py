
import pandas as pd

from LossFunction import logistic_cost_function,BinaryCrossentropy
from Neuron import Neuron
from myTensorFlow import fit
import numpy as np
from ActivationFunction import sigmoid

df = pd.read_csv("../data/student_exam_pass_dataset.csv")
X_train = np.array(df.drop(columns=["student_id", "passed"]))[:170]
X_test = np.array(df.drop(columns=["student_id", "passed"]))[-30:]
y_train = np.array(df["passed"])[:170]
y_test = np.array(df["passed"])[-30:]
m, n = X_train.shape
myNeuron = Neuron(n,sigmoid)
fit(myNeuron, X_train, y_train, learning_rate=1e-4,loss=BinaryCrossentropy)
w,b = myNeuron.get_weight()
print(f"{w}, {b}, cost: {logistic_cost_function(X_train, y_train, w, b)}")

threshold = 0.5
err = 0
for i in range(len(X_test)):
    target = y_test[i]
    predict = 1 if myNeuron.predict(X_train[i]) > threshold else 0
    err += abs(target - predict)

print(err)