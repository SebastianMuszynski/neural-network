import numpy as np


class QuadraticCost:
    @staticmethod
    def fn(y, a):
        return 0.5 * np.linalg.norm(y - a) ** 2

    @staticmethod
    def fn_d(y, a, z, a_fn):
        return (a - y) * a_fn.fn_d(z)
