import numpy as np 

def MSE(y_pred, y_true):
    """
        Mean Squared Error

        parameters : 
        -> y_true, y_pred (numpy arrays)

        returns :
        -> scalar value of the loss
    """
    return ((y_true - y_pred)**2).mean()

