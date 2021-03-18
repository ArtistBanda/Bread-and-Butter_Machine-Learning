import numpy as np
from bnbML.Utils import LossFunctions
from bnbML.Deep_Learning import ActivationFunctions
from bnbML.Deep_Learning import BackwardActivationFucntions
import progressbar


class NeuralNetwork(object):
    """
    Neural network class which tries to minimise the loss according to the optimiser
    and takes in different layers from layers module
    """

    def __init__(self):
        self.optimizer = None
        self.parameters = {}
        self.layers = []
        self.loss = None
        self.metrics = None

    def fit(self, X, y, learning_rate, epochs):
        """
        It applies the optmizer on the given dataset which first propogates in forward direction
        and then do backprop for then updating the weights associated
        """
        for _ in progressbar.progressbar(range(1, epochs)):
            # each epoch is a complete forward pass and complete backward pass for values of X
            # !!! Need to add optimizers !!!
            A, caches = self.forward_propagator(X)
            grads = self.backward_propagator(A, y, caches)
            self._update_parameters(grads, learning_rate)
        print(LossFunctions.CrossEntropyLoss(y, A))

    def add(self, layer):
        """
        Function to add layer from layers module
        """
        self.layers.append(layer)

    def compile(self, optimizer='sgd', loss='MSE', metrics=['accuracy']):
        """
        Compile function is used before fitting which is required to define the
        optimizer, loss and metrics which are required for inference
        """
        self._initialize_parameters()
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def predict(self, X):
        """
        Used to calculate the output generated when input goes through the forward propagator
        """
        A, _ = self.forward_propagator(X)
        return A

    def forward_propagator(self, X):
        """
        Used to propagate forward through the layers and calculate the output alongside cache 
        which is required for backward pass
        """
        L = len(self.layers)
        caches = []
        A = X

        for x in range(L):
            A_prev = A
            A, cache = self.layers[x].forward_pass(A_prev)
            caches.append(cache)

        return A, caches

    def backward_propagator(self, AL, Y, caches):
        """
        Used to propagate in backward direction for calculating gradients using layers derivatives
        which is required for updating parameters
        """
        L = len(self.layers)
        Y = Y.reshape(AL.shape)
        grads = {}

        dA_prev = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        for x in range(L - 1, -1, -1):
            dA_prev, dW, db = self.layers[x].backward_pass(dA_prev, caches[x])
            grads['dA' + str(x)] = dA_prev
            grads['dW' + str(x)] = dW
            grads['db' + str(x)] = db

        return grads

    def _initialize_parameters(self):
        """
        Calls parameter initialising function for all the layers
        """
        units_prev = self.layers[0].input_shape
        for i in range(1, len(self.layers)):
            units_prev = self.layers[i].initialize_parameters(units_prev)

    def _update_parameters(self, grads, learning_rate):
        """
        Updates all the parameters for all the layers
        """
        L = len(self.layers)

        for x in range(L):
            self.layers[x].update_parameters(
                grads['dW' + str(x)], grads['db' + str(x)], learning_rate)

    def _compute_cost(self, loss):
        raise NotImplementedError()

    def score(self):
        raise NotImplementedError()
