import numpy as np
from bnbML.Utils import LossFunctions
from bnbML.Deep_Learning import ActivationFunctions


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
                            str(i)] = np.random.randn(self.layers[i][1], self.layers[i - 1][1])
            self.parameters['b' + str(i)] = np.zeros((self.layers[i][1], 1))

    def _forward_pass(self, A, W, b):
        Z = np.dot(W, A) + b
        return Z

    def _forward_activation_pass(self, A_prev, W, b, activation):
        activation = getattr(ActivationFunctions, activation)
        Z = self._forward_pass(A_prev, W, b)
        A = activation(Z)
        return A

    def _backward_pass(self):
        pass

    def _compute_cost(self):
        pass

    def score(self):
        pass
