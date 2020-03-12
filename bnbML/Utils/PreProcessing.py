import numpy as np


def normalize(x):

    norms = np.abs(x).sum(axis=1)
    x /= norms[:, np.newaxis]

    return x


def train_test_split(X, y, split=0.8):
    """
        Splits the data into train and test splits

        parameters :
        -> X, y

        returns:
        -> (X_train, y_train, X_test, y_test)

    """
    length = int(len(y.shape[0]) * split)

    X_train = X[:length]
    y_train = y[:length]

    X_test = X[length:]
    y_test = y[length:]

    return (X_train, y_train, X_test, y_test)
