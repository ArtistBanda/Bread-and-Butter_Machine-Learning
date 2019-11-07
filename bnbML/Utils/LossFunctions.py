import numpy as np


def MSE(y_true, y_pred):
    """
        Mean Squared Error

        parameters : 
        -> y_true, y_pred (numpy arrays)

        returns :
        -> scalar value of the loss
    """
    return (np.square(y_true - y_pred)).mean()


def MAE(y_true, y_pred):
    """
        Mean Absolute Error

        parameters : 
        -> y_true, y_pred (numpy arrays)

        returns :
        -> scalar value of the loss
    """
    return np.abs(y_true - y_pred).mean()


def CrossEntropyLoss(y_true, y_pred, epsilon=1e-10):
    """
        Cross Entropy Loss

        parameters : 
        -> y_true, y_pred, epsilon

        returns :
        -> scalar value of the loss
    """
    predictions = np.clip(y_pred, epsilon, 1. - epsilon)
    N = y_pred.shape[0]
    ce_loss = -np.sum(np.sum(y_true * np.log(predictions + 1e-5)))/N
    return ce_loss
