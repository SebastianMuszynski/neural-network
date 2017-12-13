import collections
import numpy as np

from fn_costs.quadratic_cost import QuadraticCost


def without_first(some_array):
    return some_array[1:]


def without_last(some_array):
    return some_array[:-1]


class Layer:
    def __init__(self, size, a_fn):
        self.size = size
        self.a_fn = a_fn


class NeuralNetwork:
    def __init__(self, inputs, targets, learning_rate, cost_fn=QuadraticCost):
        self.inputs = inputs
        self.targets = targets
        self.learning_rate = learning_rate
        self.cost_fn = cost_fn

        self.layers = []
        self.weights = []
        self.biases = []

    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError('The added layer is not an instance of the Layer class.')

        self.layers.append(layer)

    def init_weights_and_biases(self):
        layer_sizes = [layer.size for layer in self.layers].insert(0, len(self.inputs))

        for n_in, n_out in zip(without_last(layer_sizes), without_first(layer_sizes)):
            self.weights.append(np.random.randn(n_out, n_in))

        for layer in without_first(self.layers):
            self.biases.append(np.random.randn(layer.size, 1))

    def feed_forward(self):
        a = self.inputs
        z_values = np.zeros(len(self.layers))
        a_values = np.zeros(len(self.layers))

        for l in range(len(self.layers)):
            layer = self.layers[l]
            w = self.weights[l]
            b = self.biases[l]

            z = np.dot(w, a) + b
            a = layer.a_fn(z)

            z_values[l] = z
            a_values[l] = a

        ff = collections.namedtuple('FeedForward', ['z_values', 'a_values'])
        return ff(z_values, a_values)

    def backpropagate(self, ff):
        deltas = np.zeros(len(self.layers))
        deltas[-1] = self.cost_fn.fn_d(y=self.targets,
                                       a=ff.a_values[-1],
                                       z=ff.z_values[-1],
                                       a_fn=self.layers[-1].a_fn)

        for l in reversed(range(len(self.layers) - 1)):
            deltas[l] = np.dot(self.weights[l + 1], deltas[l + 1]) * self.layers[l].a_fn.fn_d(ff.z_values[l])

        return deltas

    def update_weights_and_biases(self, ff, deltas):
        for l in range(len(self.layers)):
            inputs = self.inputs if l - 1 < 0 else ff.a_values[l - 1]
            self.weights[l] -= self.learning_rate * np.dot(inputs, deltas[l])
            self.biases[l] -= self.learning_rate * deltas[l]

    def fit(self):
        ff = self.feed_forward()
        deltas = self.backpropagate(ff)
        self.update_weights_and_biases(ff, deltas)
