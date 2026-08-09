import pandas as pd
import numpy as np

from ActivationFunction import relu, sigmoid
from Dense import Dense
from LossFunction import BinaryCrossentropy, logistic_cost_function
from Sequential import Sequential
df = pd.read_csv("../data/student_exam_pass_dataset.csv")
X = np.array(df.drop(columns=["student_id","passed"]))
y = np.array(df["passed"])
m,n = X.shape

model = Sequential ([
    Dense(input_size=n,units=n,activation=sigmoid,loss=BinaryCrossentropy),
    Dense(input_size=n,units=n,activation=sigmoid,loss=BinaryCrossentropy),
    Dense(input_size=n, units=1,activation=sigmoid, loss=BinaryCrossentropy)
])

model.fit(X, y, epochs=100)

print()
