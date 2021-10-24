"""
linear regression for two numpy 1d numpy arrays
(just optimizing an 2d function f(x) = m*x + t)
"""

import numpy as np
from matplotlib import pyplot as plt


class EasyLinearRegression:

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

        self.a = 0
        self.b = 1

    def fit(self):
        # numpy 1d arrays X and Y
        x = self.X
        y = self.y

        size = x.size

        if size != y.size:
            print('x.size != y.size')
            return

        # mean
        x_m, y_m = x.mean(), y.mean()
        # standard deviation (squared error / distance)
        s_x = np.sqrt(((x - x_m) ** 2).sum() / size)  # = x.std()
        s_y = np.sqrt(((y - y_m) ** 2).sum() / size)

        # this would be the error (distance) not squared
        # s_x = np.abs(x - x_m).sum() / size
        # s_y = np.abs(y - y_m).sum() / size

        # covariance
        s_xy = ((x - x_m) * (y - y_m)).sum() / size  # = np.cov(x, y);
        # correlation
        r_xy = s_xy / (s_x * s_y)

        # the function values
        # ŷ = b * x + a
        # --> a = -b * x + ŷ (replaced with x_m and y_m)
        # multiplying gradient with correlation
        self.b = (s_y / s_x) * r_xy
        self.a = -self.b * x_m + y_m


if __name__ == '__main__':
    points = 500

    # generate data
    m, c = 2, 3

    X = np.random.rand(points)
    noise = np.random.randn(points) / 4

    y = X * m + c + noise

    plt.scatter(X, y, alpha=0.6)

    reg = EasyLinearRegression(X, y)
    reg.fit()

    x_axis_points = np.linspace(0, 1, 501)
    y_axis_points = reg.b * x_axis_points + reg.a
    plt.plot(x_axis_points, y_axis_points, c='red')

    plt.show()
