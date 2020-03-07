import numpy as np


class NaiveBayesClassifier(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.parameters = []
        self.classes = None

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.classes = np.unique(self.y)
        self.parameters = []
        for i, c in enumerate(self.classes):
            X_where_c = self.X[np.where(self.y == c)]
            self.parameters.append([])
            for col in X_where_c.T:
                mean_var_dict = {
                    'mean': col.mean(), 'var': col.var()}
                self.parameters[i].append(mean_var_dict)
        print(self.parameters)

    def predict(self, X, y):
        pass

    def _calculateProbab(self):
        pass
