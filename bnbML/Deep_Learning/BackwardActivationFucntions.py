import numpy as np
from bnbML.Deep_Learning.ActivationFunctions import Sigmoid


def ReLU_backward(dA, cache):
    z = cache
    relu_der = np.where(z > 0, 1, 0)
    return dA * relu_der


def Sigmoid_backward(dA, cache):
    z = cache
    signmoid_der = Sigmoid(z) * (1 - Sigmoid(z))
    return dA * signmoid_der
