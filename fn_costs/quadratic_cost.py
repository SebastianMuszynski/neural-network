import numpy as np


class QuadraticCost:
    @staticmethod
    def fn(targets, predictions):
        return 0.5 * np.linalg.norm(targets - predictions) ** 2
