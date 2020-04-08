import numpy as np


def Sigmoid(z):
    """
        Sigmoid activation fucntion

        parameters : 
        -> z (numpy array)

        returns :
        -> numpy array with shape(z) 
    """
    return (1 / (1 + np.exp(-z)), z)


def ReLU(z):
    """
        ReLU(Rectified Linear Unit) activation fucntion

        parameters :
        -> z (numpy array)

        returns:
        -> numpy array with shape(z)
    """
    return (np.where(z > 0, z, 0), z)


def tanh(z):
    """
        tanh activation function

        parameters :
        -> z (numpy array)

        returns :
        -> numpy array with shape(z)
    """
    return (np.tanh(z), z)
