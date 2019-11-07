import numpy as np
from bnbML.Utils import LossFunctions
import matplotlib.pyplot as plt


class LinearRegression(object):
    def __init__(self):
        self.intercept = None
        self.slope = None
        self.history = []
        self.iter_count = 0

    def fit(self, x_train, y_train, epochs=10, learning_rate=0.01):
        weights = self._initialize_weights(x_train.shape)

        for x in range(epochs):

            y_cap = np.dot(x_train, weights['slope']) + weights['intercept']

            self.history.append(LossFunctions.MSE(y_train, y_cap))

            print('Loss at iter ' + str(x + 1) + ' : ' +
                  str(self.history[self.iter_count]))
            self.iter_count += 1

            N = x_train.shape[1]
            slope = (1 / N) * (np.dot(x_train.T,
                                      (np.dot(x_train, weights['slope']) - y_train)))
            intercept = (1 / N) * (np.dot(x_train, weights['slope']) - y_train)

            weights['slope'] = weights['slope'] - learning_rate * slope
            weights['intercept'] = weights['intercept'] - \
                learning_rate * intercept

        self.intercept = weights['intercept']
        self.slope = weights['slope']

    def predict(self, x):
        return np.dot(x, self.slope) + self.intercept

    def score(self, x, y):
        y_cap = np.dot(x, self.slope) + self.intercept
        return LossFunctions.MSE(y, y_cap)

    def plotLossGraph(self):
        plt.figure()
        plt.plot(np.arange(self.iter_count), self.history)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Loss')
        plt.show()

    def _initialize_weights(self, input_shape):
        slope = np.zeros((input_shape[1], 1))
        intercept = 0
        return {'slope': slope, 'intercept': intercept}
