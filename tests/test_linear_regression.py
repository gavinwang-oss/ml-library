import numpy as np
from models.linear_regression import LinearRegression

X = np.random.rand(100, 1)
y = 4 * X[:, 0] + 3

linModel = LinearRegression(0.01, 10000)

linModel.fit(X, y)

print(linModel.weights)
assert np.allclose(linModel.weights, [3, 4], atol=0.1)


