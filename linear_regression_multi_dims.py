"""
linear regression for two numpy 1d numpy arrays
(just optimizing an 2d function f(x) = m*x + t)
"""

import numpy as np


class MultivariateLinearRegression:

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X  # array of shape (n, m); where n is the number of features and m the number of rows)
        self.y = y  # 1 dim y array

    def fit(self):
        X = self.X
        y = self.y

        size = y.size

        if size != X.shape[1].size:
            print('X.shape[1].size != y.size')
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
        s_xy = ((x - x_m) * (y - y_m)).sum() / size  # = np.cov(x, y)
        # correlation
        r_xy = s_xy / (s_x * s_y)

        # the function values
        # ŷ = b * x + a
        # --> a = -b * x + ŷ (replaced with x_m and y_m)
        # multiplying gradient with correlation
        self.b = (s_y / s_x) * r_xy
        self.a = -self.b * x_m + y_m


if __name__ == '__main__':
    pass
