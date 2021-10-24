"""
test out the MultipleLinearRegression class
with multiple dimensions
"""
import numpy as np
from matplotlib import pyplot as plt

from linear_regression_multi_dims import MultivariateLinearRegression


def main():
    points = 500

    # generate data
    m, c = 2, 3

    # create random data and put them into correct shape
    X = np.random.rand(points)
    noise = np.random.randn(points) / 4

    y = X * m + c + noise

    reg = MultivariateLinearRegression(X.reshape(1, -1), y)
    reg.fit()

    print(reg.W)
    x_axis_points = np.linspace(0, 1, 501)
    y_axis_points = reg.W[0] * x_axis_points + reg.w0


if __name__ == '__main__':
    main()