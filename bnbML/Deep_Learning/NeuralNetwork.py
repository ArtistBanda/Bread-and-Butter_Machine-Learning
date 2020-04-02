import numpy as np
from bnbML.Utils import LossFunctions


class NeuralNetwork(object):
    def __init__(self, optimizer='sgd'):
        self.optimizer = optimizer
        self.parameters = None

    def fit(self, X, y, learning_rate, epochs):
        pass

    def add(self, value, layer, activation):
        activation = getattr(LossFunctions, activation)

    def _forward_pass(self):
        pass

    def _backward_pass(self):
        pass

    def score(self):
        pass
