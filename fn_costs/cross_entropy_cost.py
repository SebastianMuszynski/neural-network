import numpy as np


class CrossEntropyCost:
    @staticmethod
    def fn(targets, predictions):
        return np.sum(np.nan_to_num(-targets * np.log(predictions) - (1 - targets) * np.log(1 - predictions)))
