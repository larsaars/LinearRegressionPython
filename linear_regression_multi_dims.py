"""
linear regression for two numpy 1d numpy arrays
(just optimizing an 2d function f(x) = m*x + t)
"""

import numpy as np
from matplotlib import pyplot as plt


class MultivariateLinearRegression:

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X  # array of shape (n, m); where n is the number of features and m the number of rows)
        self.y = y  # 1 dim y array

        self.W = [0 for _ in range(X.shape[0])]  # the weights that are to be calculated (w_1 to w_n)
        self.w0 = 0  # the first weight without an x factor

    def fit(self):
        X = self.X
        y = self.y

        size = y.size
        feature_count = X.shape[0]

        if size != X.shape[1]:
            print('X.shape[1].size != y.size')
            return

        # mean of y
        y_m = y.mean()
        # y std deviation
        s_y = y.std()

        # store all X_m values
        X_m = []

        # for all axis pre calculate the parameters needed
        for axis_index in range(feature_count):
            # store the x_m value
            X_m.append(X[axis_index].mean())
            # x std
            s_x = X[axis_index].std()
            # covariance
            s_xy = ((X[axis_index] - X_m[axis_index]) * (y - y_m)).sum() / size
            # correlation
            r_xy = s_xy / (s_x * s_y)
            # set the weight
            self.W[axis_index] = (s_y / s_x) * r_xy

        # w0 = ŷ - w_1*x_1 - ... - w_n * x_n
        # x_1, ..., x_n is being replaced by x_m at the correct position,
        # same as ŷ is replaced by y_m in this formula
        self.w0 = y_m

        for axis_index in range(feature_count):
            self.w0 -= self.W[axis_index] * X_m[axis_index]

    def solve(self, X_values: list):
        if(len(X_values) != len(self.W)):
            print('X_values length does not match the expected size')
            return None

        y_d = self.w0

        for i in range(len(self.W)):
            y_d += self.W[i] * X_values[i]

        return y_d



if __name__ == '__main__':
    points = 500

    # generate data
    m, c = 2, 3

    # create random data and put them into correct shape
    X = np.random.rand(points)
    noise = np.random.randn(points) / 4

    y = X * m + c + noise

    plt.scatter(X, y, alpha=0.6)

    reg = MultivariateLinearRegression(X.reshape(1, -1), y)
    reg.fit()

    print(reg.W)
    x_axis_points = np.linspace(0, 1, 501)
    y_axis_points = reg.W[0] * x_axis_points + reg.w0
    plt.plot(x_axis_points, y_axis_points, c='red')

    plt.show()
