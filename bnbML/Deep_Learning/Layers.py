import numpy as np
from bnbML.Deep_Learning import ActivationFunctions, BackwardActivationFucntions


class Layer(object):
    def forward_pass(self, A_prev):
        raise NotImplementedError

    def backward_pass(self, dA, caches):
        raise NotImplementedError

    def update_parameters(self):
        pass


class Input(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward_pass(self, A_prev):
        return A_prev, None

    def backward_pass(self, dA_prev, cache):
        return dA_prev, None, None

    def update_parameters(self, dW, db, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, units, activation='Linear'):
        self.units = units
        self.activation = activation
        self.parameters = {}

    def initialize_parameters(self, prev_units):
        self.parameters['W'] = np.random.rand(self.units, prev_units)
        self.parameters['b'] = np.zeros((self.units, 1))
        return self.units

    def forward_pass(self, A_prev):
        activation = getattr(ActivationFunctions, self.activation)

        Z, linear_cache = self._forward_pass_helper(A_prev)
        A, activation_cache = activation(Z)

        caches = (linear_cache, activation_cache)
        return A, caches

    def _forward_pass_helper(self, A_prev):
        cache = (A_prev, self.parameters['W'], self.parameters['b'])
        Z = np.dot(self.parameters['W'], A_prev) + self.parameters['b']

        return Z, cache

    def backward_pass(self, dA, caches):
        linear_cache, activation_cache = caches
        backward_activation = getattr(
            BackwardActivationFucntions, self.activation + '_backward')

        dZ = backward_activation(dA, activation_cache)
        dA_prev, dW, db = self._backward_pass_helper(dZ, linear_cache)

        return dA_prev, dW, db

    def _backward_pass_helper(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return (dA_prev, dW, db)

    def update_parameters(self, dW, db, learning_rate):
        self.parameters['W'] -= learning_rate * dW
        self.parameters['b'] -= learning_rate * db

# In progress
# not working yet


class Conv1D(Layer):
    def __init__(self, size, units, stride=1, padding=0, activation='ReLU'):
        """
        size -> It is the size of the filter, eg : [1, 1] , then size = 2

        units -> This is the number of filters in the layer, eg : [1, 1], [1, 1] , then untis = 2

        Stride is the size of swipe it will take per operation

        Padding is the number of values added to the left and right of the layer    

        """
        self.units = units
        self.size = size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.parameters = None

    def initialize_parameters(self):
        """
        It initializes the value of parameters with the size of (untis, size)
        """
        self.parameters = np.random.rand(self.units, self.size)

    def forward_pass(self, A_prev):
        """
        It is one step for forward propagation of a CONV1D layer

        """
        activation = getattr(ActivationFunctions, self.activation)

        # Creates the boiler plate for the output to be produced with the size of
        # (units, n - s + 1)
        # where n is the number of elements per channel in the input
        Z = np.zeros((self.units, A_prev.shape[1] - self.size + 1, ))

        for unit in range(self.units):
            # Iterating through all the multiplication checkpoints per filter of the conv1D layer
            for A_prev_pos in range(0, A_prev.shape[1] - self.size + 1, self.stride):

                # Inserting the value per unit for the particular multiplication checkpoint
                # Eg : np.dot([[1, 2], [2, 4]],[1, 2]) = [5, 10] Then summing over the output using np.sum()
                Z[unit, A_prev_pos] = np.sum(
                    np.dot(A_prev[:, A_prev_pos:A_prev_pos + self.size], self.parameters[unit]))

        A, activation_cache = activation(Z)
        linear_cache = A_prev, self.parameters

        caches = (linear_cache, activation_cache)

        return A, caches

    def backward_pass(self):
        pass
