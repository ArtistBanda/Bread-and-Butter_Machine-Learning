import numpy as np
from bnbML.Deep_Learning import ActivationFunctions
from bnbML.Utils import LossFunctions
from bnbML.Utils.Metrics import accuracy
from bnbML.Utils.Plotting import plotLossGraph


class BinaryLogisticRegression(object):
    def __init__(self):
        self.weights = None
        self.bias = 0
        self.history = []
        self.iter_count = 0

    def fit(self, x_train, y_train, epochs=10, learning_rate=0.01):
        weights = self._initialize_weights(x_train.shape)

        for x in range(epochs):

            y_cap = ActivationFunctions.Sigmoid(
                np.dot(x_train, weights['weights']) + weights['bias'])

            self.history.append([LossFunctions.CrossEntropyLoss(
                y_train, y_cap), accuracy(y_train, y_cap)])

            print('Loss at iter ' + str(x + 1) + ' : ' + str(self.history[self.iter_count][0]) + '  Accuracy at ' + str(
                x + 1) + ' : ' + str(self.history[self.iter_count][1]))
            self.iter_count += 1

            N = x_train.shape[1]

            grad_w = (1 / N) * np.dot(x_train.T, (y_cap - y_train))
            grad_b = (1 / N) * np.sum(y_cap - y_train)

            weights['weights'] = weights['weights'] - learning_rate * grad_w
            weights['bias'] = weights['bias'] - learning_rate * grad_b

        self.weights = weights['weights']
        self.bias = weights['bias']

    def predict(self, x):
        return ActivationFunctions.Sigmoid(np.dot(x, self.weights) + self.bias)

    def plotLossGraph(self):
        plotLossGraph(self.history, self.iter_count)

    def _initialize_weights(self, input_shape):
        weights = np.zeros((input_shape[1], 1))
        bias = 0
        return {'weights': weights, 'bias': bias}
