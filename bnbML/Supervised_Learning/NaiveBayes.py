import numpy as np


class NaiveBayesClassifier(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.parameters = []
        self.class_proba = []
        self.classes = None

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.classes = np.unique(self.y)
        self.parameters = []
        for i, c in enumerate(self.classes):
            X_where_c = self.X[np.where(self.y == c)]
            self.class_proba.append(X_where_c.shape[0] / self.X.shape[0])
            self.parameters.append([])
            for col in X_where_c.T:
                mean_var_dict = {
                    'mean': col.mean(), 'var': col.var()}
                self.parameters[i].append(mean_var_dict)

    def predict(self, X):
        y_pred = []
        for i, x in enumerate(X):
            y_pred.append([])
            for j, _ in enumerate(self.classes):
                y_pred[i].append(1)
                for k, _ in enumerate(self.parameters):
                    y_pred[i][j] *= self._conditionalProba(j, k, x)
                y_pred[i][j] /= self.class_proba[j]
            y_pred[i] = self.classes[np.argmax(y_pred[i])]
        return y_pred

    def score(self, X, y):
        pass

    def _conditionalProba(self, c, feature, x):
        eps = 1e-4
        mean = self.parameters[c][feature]['mean']
        var = self.parameters[c][feature]['var']
        coeff = 1 / (np.sqrt(2 * np.pi * np.power(var, 2)) + eps)
        exponent = np.exp(-np.power(x - mean, 2) /
                          (2 * np.power(var, 2) + eps))
        return np.sum(coeff * exponent)
