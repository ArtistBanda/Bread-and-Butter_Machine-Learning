import numpy as np
from bnbML.Deep_Learning import ActivationFunctions


def accuracy(y_true, y_pred, threshold=0.5):
    prediction = np.where(y_pred > threshold, 1, 0)
    correct = np.where(prediction == y_true, 1, 0)
    acc = (np.sum(correct) / y_true.shape[0]) * 100
    return acc


def mean_accuracy(y_true, y_pred):
    acc = np.sum(np.where(y_true == y_pred, 1, 0)) / y_true.shape[0]
    return acc * 100
