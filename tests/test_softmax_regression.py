import numpy as np
from models.softmax_regression import SoftmaxRegression

X = np.random.rand(100, 1)
y = (X[:, 0] * 3).astype(int)

model = SoftmaxRegression(0.01, 10000)

model.fit(X, y)

predictions = model.predict(X)

accuracy = np.mean(predictions == y)

print(accuracy)

assert accuracy > 0.9



