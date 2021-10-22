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
        feature_size = X.shape[0]

        if size != X.shape[1].size:
            print('X.shape[1].size != y.size')
            return

        # mean
        X_m, y_m = X.mean(), y.mean()
        # y std deviation
        s_y = y.std()

        # for all axis pre calculate the parameters needed
        for axis_index in range(feature_size):
            # x std
            s_x = X[axis_index].std()
            # covariance
            s_xy = np.cov(X[axis_index], y)
            # correlation
            r_xy = s_xy / (s_x * s_y)

        #TODO
        # the function values
        # ŷ = b * x + a
        # --> a = -b * x + ŷ (replaced with x_m and y_m)
        # multiplying gradient with correlation
        self.b = (s_y / s_x) * r_xy
        self.a = -self.b * x_m + y_m


if __name__ == '__main__':
    pass
