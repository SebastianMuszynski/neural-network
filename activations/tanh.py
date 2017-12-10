from activations.sigmoid import Sigmoid


class Tanh:
    @staticmethod
    def fn(x):
        return Sigmoid.fn(2 * x) * 2 - 1

    @staticmethod
    def fn_d(x):
        return 1 - Tanh.fn(x) ** 2
