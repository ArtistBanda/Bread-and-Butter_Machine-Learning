import numpy as np
from bnbML.Utils.Metrics import mean_accuracy 


class NaiveBayesClassifier(object):
    
    """
        A Naive Bayes classifier is a probabilistic machine learning
        model that’s used for classification task.
        The crux of the classifier is based on the Bayes theorem.
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.parameters = []
        self.class_proba = []
        self.classes = None

    def fit(self, X_train, y_train):
        """
            Fit fucntion which sets the parameters for the classifier according 
            too the given data.

            parameters :
            -> X_train, y_train (numpy arrays)

            returns :
            -> void

        """
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
        """
            Predict function calculates the y_pred array
            for the passed X from the previously calculated
            and stored parameters.

            parameters :
            -> X

            returns :
            -> y_pred

        """
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

    def accuracy(self, X, y):
        """
            Calculates mean accuracy

            parameters:
            -> X, y

            returns:
            -> accuracy (in percentage)

        """
        y_pred = self.predict(X)
        return mean_accuracy(y, y_pred)

    def _conditionalProba(self, c, feature, x):
        """
            (helper function)
            Calculates conditional probability for the given
            class and feature for a single input data

            parameters:
            -> c, feature, x

            returns:
            -> conditional probability (0-1)

        """
        eps = 1e-4
        mean = self.parameters[c][feature]['mean']
        var = self.parameters[c][feature]['var']
        coeff = 1 / (np.sqrt(2 * np.pi * np.power(var, 2)) + eps)
        exponent = np.exp(-np.power(x - mean, 2) /
                          (2 * np.power(var, 2) + eps))
        return np.sum(coeff * exponent)
