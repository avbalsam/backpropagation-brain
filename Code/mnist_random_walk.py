"""
mnist_random_walk.py
~~~~~~~~~~

Implements attention for MNIST task by slowly moving the image across the screen.
This allows us to test the effectiveness of attention on distinct tasks.

Author: Avi Balsam
"""

#### Libraries
# Standard library
import os
import random

# Third-party libraries
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import relplot

import mnist_loader


def relu(z):
    return np.maximum(z, z * 0.01) + 1


def relu_prime(z):
    res = relu(z)
    res[res > 0] = 1
    res[res < 0] = 0.01
    res[res == 0] = 0.01
    return res


class Network(object):
    activation_func = lambda x, y: sigmoid(y)
    activation_func_deriv = lambda x, y: sigmoid_prime(y)

    # activation_func = lambda x, y: relu(y)
    # activation_func_deriv = lambda x, y: relu_prime(y)

    def __init__(self, sizes, time_delay, max_offset):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.error_queue = [np.zeros(sizes[-1]) for _ in range(time_delay)]
        self.max_offset = max_offset

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activation_func(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        test_data = list(test_data)
        training_data = list(training_data)

        error_history = list()
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            training_data_transformed = []
            for x, y in training_data:
                two_d = np.reshape(x, (28, 28))
                for offset in range(self.max_offset):
                    offset_mat = np.hstack((np.zeros((28, offset)), two_d))
                    frame = np.delete(offset_mat, list(range(28, offset_mat.shape[1])), axis=1)
                    training_data_transformed.append((frame.reshape((784, 1)), y))

            mini_batches = [
                training_data_transformed[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                error = self.evaluate(test_data)
                print("Epoch {0}: {1} / {2}".format(
                    j, error, n_test))
                error_history.append([j, error])
                df = pd.DataFrame(error_history, columns=["Epoch", "# Correct"])
                df.to_csv(f'./{DIR_NAME}/error_history.csv')
                g = relplot(df, x="Epoch", y="# Correct", kind="line")
                g.savefig(f"./{DIR_NAME}/error_history.png")
                plt.close()
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop_attention(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop_attention(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_func(z)
            activations.append(activation)
        # backward pass
        self.error_queue.append(
            self.cost_derivative(activations[-1], y) *
            self.activation_func_deriv(zs[-1])
        )
        delta = self.error_queue.pop(0)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_func_deriv(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == '__main__':
    # training_data, validation_data, test_data = \
    #    chessboard_loader.load_data_wrapper()
    #
    # net = Network([2, 30, 2])
    # net.SGD(training_data, 60, 10, .1, test_data=test_data)

    for delay in range(0, 26):
        for iteration in range(25):
            training_data, validation_data, test_data = \
                mnist_loader.load_data_wrapper()

            print(f"Starting model for tau={delay}, iteration: {iteration}...")
            start = time.time()
            ERROR_TIME_DELAY = delay

            # Base dir name to use. Each network will add on to this.
            DIR_NAME = f"mnist_attention/time_delay_tests_error_dependent_iterations/chessboard_error_time_delay:{ERROR_TIME_DELAY}/iteration_{iteration}"

            net = Network([784, 100, 10], time_delay=delay, max_offset=14)
            if os.path.exists(DIR_NAME):
                print(f"Model {DIR_NAME} already exists. Skipping...")
                continue
            try:
                os.makedirs(DIR_NAME)
            except FileExistsError:
                print(f"Model {DIR_NAME} already exists. Skipping...")
                continue

            net.SGD(training_data, 60, 100, 3, test_data=test_data)
            print(
                f"Finished model for tau={delay}, iteration: {iteration}\nTime elapsed: {time.time() - start}\n\n\n\n"
            )
