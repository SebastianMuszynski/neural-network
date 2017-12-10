import numpy as np


class Sigmoid:
    @staticmethod
    def fn(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def fn_d(x):
        y = Sigmoid.fn(x)
        return y * (1 - y)
