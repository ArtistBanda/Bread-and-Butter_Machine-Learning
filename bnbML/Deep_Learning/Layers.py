import numpy as np
from bnbML.Deep_Learning import ActivationFunctions, BackwardActivationFucntions


class Layer(object):
    def forward_pass(self, A_prev):
        raise NotImplementedError

    def backward_pass(self, dA, caches):
        raise NotImplementedError

    def update_parameters(self):
        pass


class Input(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward_pass(self, A_prev):
        return A_prev, None

    def backward_pass(self, dA_prev, cache):
        return dA_prev, None, None

    def update_parameters(self, dW, db, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, units, activation='Linear'):
        self.units = units
        self.activation = activation
        self.parameters = {}

    def initialize_parameters(self, prev_units):
        self.parameters['W'] = np.random.rand(self.units, prev_units)
        self.parameters['b'] = np.zeros((self.units, 1))
        return self.units

    def forward_pass(self, A_prev):
        activation = getattr(ActivationFunctions, self.activation)

        Z, linear_cache = self._forward_pass_helper(A_prev)
        A, activation_cache = activation(Z)

        caches = (linear_cache, activation_cache)
        return A, caches

    def _forward_pass_helper(self, A_prev):
        cache = (A_prev, self.parameters['W'], self.parameters['b'])
        Z = np.dot(self.parameters['W'], A_prev) + self.parameters['b']

        return Z, cache

    def backward_pass(self, dA, caches):
        linear_cache, activation_cache = caches
        backward_activation = getattr(
            BackwardActivationFucntions, self.activation + '_backward')

        dZ = backward_activation(dA, activation_cache)
        dA_prev, dW, db = self._backward_pass_helper(dZ, linear_cache)

        return dA_prev, dW, db

    def _backward_pass_helper(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return (dA_prev, dW, db)

    def update_parameters(self, dW, db, learning_rate):
        self.parameters['W'] -= learning_rate * dW
        self.parameters['b'] -= learning_rate * db

# In progress

# class Conv2D(Layer):
#     def __init__(self):
#         pass

#     def initialize_parameters(self):
#         pass

#     def forward_pass(self):
#         pass

#     def backward_pass(self):
#         pass
