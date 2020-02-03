import numpy as np
from bnbML.Utils import LossFunctions
from bnbML.Utils.Plotting import plotLossGraph
from bnbML.Utils.PreProcessing import normalize
import matplotlib.pyplot as plt
import pandas as pd
import progressbar


class LinearRegression(object):
    def __init__(self, normalize=False):
        self.weights = None
        self.history = []
        self.iter_count = 0
        self.normalize = normalize

    def fit(self, x_train, y_train, epochs=10, learning_rate=0.01):

        x_train, y_train = self._checks(x_train, y_train)

        x_train = self._insert_bias(x_train)

        if self.weights is None:
            self.weights = self._initialize_weights(x_train.shape)

        for _ in progressbar.progressbar(range(epochs)):

            y_cap = np.dot(x_train, self.weights)

            self.history.append(LossFunctions.MSE(y_train, y_cap))

            print("Loss : " + str(self.history[len(self.history) - 1]))

            self.iter_count += 1

            N = x_train.shape[0]
            slope = (1 / N) * (np.dot(x_train.T,
                                      (np.dot(x_train, self.weights) - y_train)))

            self.weights = self.weights - learning_rate * slope

    def predict(self, x):
        x = self._insert_bias(x)
        return np.dot(x, self.weights)

    def score(self, x, y):
        x = self._insert_bias(x)
        x, y = self._checks(x, y)
        y_cap = np.dot(x, self.weights)
        return LossFunctions.MSE(y, y_cap)

    def plotLossGraph(self):
        plotLossGraph(self.history, self.iter_count)

    def _initialize_weights(self, input_shape):
        return np.zeros((input_shape[1], 1))

    def _insert_bias(self, x):
        return np.insert(x, 0, 1, axis=1)

    def _checks(self, x, y):
        if y.shape is not tuple((len(y), 1)):
            y = np.reshape(y, (y.shape[0], 1))

        if self.normalize is True:
            x = normalize(x)

        return x, y
