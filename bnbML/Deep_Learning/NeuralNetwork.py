import numpy as np
from bnbML.Utils import LossFunctions
from bnbML.Deep_Learning import ActivationFunctions
from bnbML.Deep_Learning import BackwardActivationFucntions


class NeuralNetwork(object):
    def __init__(self):
        self.optimizer = None
        self.parameters = {}
        self.layers = []
        self.loss = None
        self.metrics = None

    def fit(self, X, y, learning_rate, epochs):
        pass

    def add(self, value, layer, activation=None):
        if activation:
            self.layers.append([layer, value, activation])
        self.layers.append([layer, value])

    def compile(self, optimizer='sgd', loss='MSE', metrics=['accuracy']):
        self._initialize_parameters()
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def _initialize_parameters(self):
        for i in range(1, len(self.layers)):
            self.parameters['W' +
                            str(i)] = np.random.randn(self.layers[i][1], self.layers[i - 1][1]) * 0.01
            self.parameters['b' + str(i)] = np.zeros((self.layers[i][1], 1))

    def _forward_pass(self, A, W, b):
        cache = (A, W, b)
        Z = np.dot(W, A) + b

        return Z, cache

    def _forward_activation_pass(self, A_prev, W, b, activation):
        activation = getattr(ActivationFunctions, activation)

        Z, linear_cache = self._forward_pass(A_prev, W, b)
        A, activation_cache = activation(Z)

        caches = (linear_cache, activation_cache)

        return A, caches

    def _backward_pass(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return (dA_prev, dW, db)

    def _backward_activation_pass(self, dA, caches, activation):
        linear_cache, activation_cache = caches
        activation = getattr(ActivationFunctions, activation)
        backward_activation = getattr(
            ActivationFunctions, activation + '_backward')

        dZ = backward_activation(dA, activation_cache)
        dA_prev, dW, db = self._backward_pass(dZ, linear_cache)

        return dA_prev, dW, db

    def _compute_cost(self):
        pass

    def score(self):
        pass
