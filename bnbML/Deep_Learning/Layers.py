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
    """
    Base class Layer from which all the layers will be derived from
    """

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward_pass(self, A_prev):
        return A_prev, None

    def backward_pass(self, dA_prev, cache):
        return dA_prev, None, None

    def update_parameters(self, dW, db, learning_rate):
        pass


class Dense(Layer):
    """
    A Dense is one of the most simple layer which is used in Neural Networks.
    It requires the number of neurons (units) it will have along the activation 
    function it will use.
    """

    def __init__(self, units, activation='Linear'):
        self.units = units
        self.activation = activation
        self.parameters = {}

    def initialize_parameters(self, prev_units):
        self.parameters['W'] = np.random.rand(self.units, prev_units)
        self.parameters['b'] = np.zeros((self.units, 1))
        return self.units

    def forward_pass(self, A_prev):
        """
        It propagates the layers in forward direction which simply means it multiplies
        the weights with the provided input, adds bias and applies activation function to it.
        """

        activation = getattr(ActivationFunctions, self.activation)

        Z, linear_cache = self._forward_pass_helper(A_prev)
        A, activation_cache = activation(Z)

        # Creates a tuple of linear and activation cache
        caches = (linear_cache, activation_cache)
        return A, caches

    def _forward_pass_helper(self, A_prev):
        """
        This is a helper function for forward pass which creates linear cache and performs the
        actual linear propogation (multiplying weights and adding bias)
        """
        cache = (A_prev, self.parameters['W'], self.parameters['b'])
        Z = np.dot(self.parameters['W'], A_prev) + self.parameters['b']

        return Z, cache

    def backward_pass(self, dA, caches):
        """
        It propagates the layer in backward direction (backpropagation) which is required to update
        the weights which is ultimately results in reducing the loss.
        """

        linear_cache, activation_cache = caches
        backward_activation = getattr(
            BackwardActivationFucntions, self.activation + '_backward')

        dZ = backward_activation(dA, activation_cache)

        # dA_prev is required for the previous layers which are behind this layer
        # dW and dB are required to update the weights
        dA_prev, dW, db = self._backward_pass_helper(dZ, linear_cache)

        return dA_prev, dW, db

    def _backward_pass_helper(self, dZ, cache):
        """
        This is a helper function for backward pass which does the actual derivative calculation
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        # calculating the derivative and then dividing it over the number of samples
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        # asserting for correct shapes and dimensions for the derivatives
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return (dA_prev, dW, db)

    def update_parameters(self, dW, db, learning_rate):
        """
        Updating the parameters to move towards the global minima of the loss
        """

        # learning rate multiplied with the derivative of the weights and biases
        self.parameters['W'] -= learning_rate * dW
        self.parameters['b'] -= learning_rate * db


class Flatten(Layer):
    """
    A layer class which reduces the previous layers output (usually a conv layer) into 
    a straight linear array of size (n, 1)
    """

    def forward_pass(self, A_prev):
        """
        The forward propagator for Flatten array which ravels/flattens the input layer
        """
        cache = A_prev

        # converts ndarray of shape (n, m, k ...) to (n+m+k+...., )
        A = A_prev.flatten()
        # converts A from shape of (n, ) to (n, 1)
        A = A.reshape((A.shape, 1))

        return A, cache

    def backward_pass(self, dA, cache):
        """
        The backward propagator for Flatten layer which unravels of flattens the
        derivative of the next layer
        """

        # reshapes (n, 1) to the shape of the cache which was the input to the
        # forward pass
        dA_prev = dA.reshape(cache.shape)

        # None is returned in the place of dW and db
        return dA_prev, None, None


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
        self.parameters = {}

    def initialize_parameters(self):
        """
        It initializes the value of parameters with the size of (untis, size)
        """
        self.parameters['W'] = np.random.rand(self.units, self.size)
        self.parameters['b'] = np.zeros((self.units, 1))

    def forward_pass(self, A_prev):
        """
        It is one step for forward propagation of a CONV1D layer
        """
        activation = getattr(ActivationFunctions, self.activation)

        # Creates the boiler plate for the output to be produced with the size of
        # (units, n - s + 1)
        # where n is the number of elements per channel in the input
        Z = np.zeros((self.units, A_prev.shape[1] - self.size + 1))

        for unit in range(self.units):
            # Iterating through all the multiplication checkpoints per filter of the conv1D layer
            for A_prev_pos in range(0, A_prev.shape[1] - self.size + 1, self.stride):

                # Inserting the value per unit for the particular multiplication checkpoint
                # Eg : np.dot([[1, 2], [2, 4]],[1, 2]) = [5, 10] Then summing over the output using np.sum()
                print(self.parameters['W'], self.parameters['b'])
                Z[unit, A_prev_pos] = np.sum(
                    np.dot(A_prev[:, A_prev_pos:A_prev_pos + self.size], self.parameters['W'][unit])) + self.parameters['b'][unit]

        A, activation_cache = activation(Z)
        linear_cache = A_prev, self.parameters

        caches = (linear_cache, activation_cache)

        return A, caches

    def backward_pass(self, dA, caches):
        """
        The backward propagator for CONV 1D layer which calculates the matrix / vector derivatives
        with respect to the incoming derivative from the next layer
        """

        # Seperating caches
        linear_cache, activation_cache = caches
        # Extracting function from the string
        backward_activation = getattr(
            BackwardActivationFucntions, self.activation + '_backward')

        # _ are parameters, will update this dependency later
        A_prev, _ = linear_cache

        # dZ is the derivative with respect to the activation function
        dZ = backward_activation(dA, activation_cache)

        dW = np.zeros((self.parameters['W'].shape))
        db = np.zeros((self.parameters['b'].shape))
        dA_prev = np.zeros((dZ.shape))

        # iterating over the number of filters / channels
        for unit in range(self.units):
            # db is the sum of dZ averaged out
            db[unit] = np.sum(dZ[unit]) / len(dZ[unit])

            # for dW performing convolution on A_prev and dZ
            for A_prev_pos in range(0, A_prev.shape[1] - len(dZ[unit]) + 1, self.stride):
                dW[unit, A_prev_pos] = np.sum(
                    np.dot(A_prev[:, A_prev_pos:A_prev_pos + len(dZ[unit])], dZ[unit]))

            dZ_padded = np.pad(dZ, (1, 1), 'constant')

            # for dA_prev performing convolution on dZ_padded and dW
            for dZ_pos in range(0, dZ_padded.shape[1] - len(self.parameters['W'][unit]), self.stride):
                dA_prev[unit] = np.sum(
                    np.dot(dZ_padded[:, dZ_pos:dZ_pos + len(self.parameters['W'][unit])], dW[unit]))

        return (dA_prev, dW, db)

    def update_parameters(self, dW, db, learning_rate):
        """
        Updating the parameters to move towards the global minima of the loss
        """

        # learning rate multiplied with the derivative of the weights and biases
        self.parameters['W'] -= learning_rate * dW
        self.parameters['b'] -= learning_rate * db
