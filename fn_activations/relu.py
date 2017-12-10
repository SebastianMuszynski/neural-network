import numpy as np


class Relu:
    @staticmethod
    def fn(x):
        return np.maximum(x, 0)

    @staticmethod
    def fn_d(x):
        return np.heaviside(x, 1)
