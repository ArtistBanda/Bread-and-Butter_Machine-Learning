import numpy as np
from bnbML.Deep_Learning.ActivationFunctions import Sigmoid


def ReLU_backward(dA, cache):
    z = cache
    relu_der = np.where(z > 0, 1, 0)
    return dA * relu_der


def Sigmoid_backward(dA, cache):
    z = cache
    sig_z, _ = Sigmoid(z)
    signmoid_der = sig_z * (1 - sig_z)
    return dA * signmoid_der
