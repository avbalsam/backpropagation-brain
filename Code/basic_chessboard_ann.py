"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

Credit: Nielsen, 2015
"""

import itertools
import math
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from seaborn import relplot

import chessboard_loader
import os

BOARD_SIZE = 30
NUM_SQUARES_PER_ROW = 3

# Base dir name to use. Each network will add on to this.
DIR_NAME = f"chessboard_pixsize:{BOARD_SIZE}_numsquares:{NUM_SQUARES_PER_ROW}"


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(z, z * 0.01) + 1


def relu_prime(z):
    res = relu(z)
    res[res > 0] = 1
    res[res < 0] = 0.01
    res[res == 0] = 0.01
    return res


def relu_exp(z):
    return relu(z) + 1


def softplus(z):
    return np.log(1 + np.exp(z))


def multilayered_graph(*subset_sizes, weights):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.Graph()
    for i, layer in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    # for layer1, layer2 in nx.utils.pairwise(layers):
    #     G.add_edges_from(itertools.product(layer1, layer2))

    # layers is a 2d array which maps each node's "position" in the multipartite layout to its nx index
    layers = [[] for _ in range(len(layers))]
    for node_index, layer in nx.get_node_attributes(G, "layer").items():
        layers[layer].append(node_index)

    for layer, weight_mat in enumerate(weights):
        for (row, col), weight in np.ndenumerate(weight_mat):
            if weight != 0:
                G.add_edge(layers[layer][col], layers[layer+1][row], weight=weight)
    return G


class Network(object):
    #activation_func = lambda x, y: sigmoid(y)
    #activation_func_deriv = lambda x, y: sigmoid_prime(y)
    #activation_func_name = "sigmoid"

    activation_func = lambda x, y: relu(y)
    activation_func_deriv = lambda x, y: relu_prime(y)
    activation_func_name = "relu"

    # activation_func = lambda x, y: relu_exp(y)
    # activation_func_deriv = lambda x, y: relu_exp(y)

    # activation_func = lambda x, y: softplus(y)
    # activation_func_deriv = lambda x, y: sigmoid(y)

    # For experimental backprop testing
    total_weight_calculations = 0
    correct_weight_calculations = 0

    def __init__(self, sizes):
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

        self.dir_name = f"{DIR_NAME}_{str(self)}"

        # We only need biases for the later layers
        self.biases = [np.random.randn(y, 1)+1 for y in sizes[1:]]

        # Each weight is a matrix with rows = output layer and columns = input layer
        # Same convention as book
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # Experimental backprop testing
        element = (0, 0)
        self.weight_grad_error = list()
        for w in self.weights:
            g = np.empty(w.shape, dtype=object)
            g.fill(element)
            self.weight_grad_error.append(g)

    def __str__(self):
        return f"Net{self.sizes}"

    def visualize(self):
        """Visualize the neurons in this network"""
        subset_sizes = self.sizes

        G = multilayered_graph(*subset_sizes, weights=self.weights)
        color = ["blue" for v, data in G.nodes(data=True)]
        pos = nx.multipartite_layout(G, subset_key="layer")
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, node_color=color, with_labels=False)
        plt.show()

    def get_output(self, a):
        """Return the output of the network if ``a`` is input."""
        zs, activations = self.feedforward(a)
        return activations[-1]

    def feedforward(self, x):
        """Runs a feedforward pass on input x"""
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            # Experimental method which adds extra bias to all the activations to keep them large
            z = np.dot(w, activation) + b
            zs.append(z)

            if i == self.num_layers - 2:
                #print(z)
                activation = sigmoid(z)
            else:
                activation = self.activation_func(z)
            activations.append(activation)
        #print(activations[-1])
        return zs, activations

    def SGD(self, tr_data, epochs, mini_batch_size, eta,
            te_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        error_history = list()

        te_data = list(te_data)
        tr_data = list(tr_data)
        if te_data: n_test = len(te_data)
        n = len(tr_data)
        for j in range(epochs):
            random.shuffle(tr_data)
            mini_batches = [
                tr_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if te_data and j % 10 == 0:
                error = self.evaluate(te_data)
                error_history.append([j, error])
                print("Epoch {0}: {1}".format(
                    j, error))

                # Testing for experimental backprop
                if self.total_weight_calculations != 0:
                    print(f"Total weight calculations: {self.total_weight_calculations}, correct: {self.correct_weight_calculations}, percent correct: {round(self.correct_weight_calculations/self.total_weight_calculations*100, 3)}%")

            if te_data and j % 100 == 0:
                if not os.path.exists(f"./{self.dir_name}"):
                    os.makedirs(f"./{self.dir_name}/representations")
                df = pd.DataFrame(error_history, columns=["Epoch", "Error"])
                g = relplot(df, x="Epoch", y="Error", kind="line")
                g.savefig(f"./{self.dir_name}/error_history.png")
                plt.close()

                # Save the chessboard as the NN currently represents it
                img_arr = np.ndarray((BOARD_SIZE, BOARD_SIZE))
                for inp, exp_result in training_data:
                    x = int(inp[0, 0])
                    y = int(inp[1, 0])
                    img_arr[x, y] = net.get_output(inp)[0, 0] * 255

                img = Image.fromarray(np.array(img_arr))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(f"./{self.dir_name}/representations/chessboard_representation_epoch:{j}.png")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs, activations = self.feedforward(x)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_func_deriv(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def backprop_weights(self, x, y):
        """Backpropagation algorithm which propagates weights instead of a global error term.
        The change in weight from each layer can be used to find the change in weight
        for the previous layer, and to find the biases for the previous layer."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs, activations = self.feedforward(x)

        # We only do the first step of the backward pass. This step only depends on the activations of the
        # output layer, and is easy for those neurons to compute.
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_func_deriv(z)

            for j in range(nabla_w[-l].shape[0]):
                # The bias vector equals the error vector
                nabla_b[-l][j, 0] = (sp[j] / activations[-l][j]) * \
                                    sum(
                                        self.weights[-l + 1].transpose()[j, k_prime] * nabla_w[-l + 1][k_prime, j]
                                        for k_prime in range(nabla_w[-l + 1].shape[0])
                                    )

                for k in range(nabla_w[-l].shape[1]):
                    # Full formula to get the weights of a layer from the weights of the next layer
                    # This formula is extremely inefficient, since it recomputes the j-th term of the
                    # error for every input neuron.
                    # nabla_w[-l][j, k] = activations[-l - 1][k] * (sp[j] / activations[-l][j]) * \
                    #                     sum(
                    #                         self.weights[-l + 1].transpose()[j, k_prime] * nabla_w[-l + 1][k_prime, j]
                    #                         for k_prime in range(nabla_w[-l + 1].shape[0])
                    #                     )

                    # To get the w_jk, we can multiply the bias (error) by the activation of the input neuron
                    nabla_w[-l][j, k] = activations[-l - 1][k] * nabla_b[-l][j, 0]

            # Alternate way of computing the bias from delta_w (this essentially recomputes the error)
            # for j in range(len(nabla_b[-l])):
            #     nabla_b[-l][j,0] = sum(
            #         nabla_w[-l][j, k] / activations[-l - 1][k,0] for k in range(len(nabla_w[-l][j])) if abs(activations[-l - 1][k,0]) > 0.00001
            #     ) / len(nabla_w[-l][j])

        return nabla_b, nabla_w

    def backprop_experimental(self, x, y):
        """Experimental method of computing nabla_w using only nabla_w from the next layer,
        and not the current activation. The derivation of this method assumes that g'(x)/g(x) = 1/x,
        where g is the activation function (this is true for ReLu, and approximately true for softplus).
        It also assumes that, for every weight, the activation of the presynaptic neuron is much greater
        than the weighted input from all other neurons."""
        # TODO: Make this work
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs, activations = self.feedforward(x)

        # We only do the first step of the backward pass. This step only depends on the activations of the
        # output layer, and is easy for those neurons to compute.
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_func_deriv(z)

            for j in range(nabla_w[-l].shape[0]):
                for k in range(nabla_w[-l].shape[1]):
                    if self.weights[-l][j, k] == 0:
                        nabla_w[-l][j, k] = 0
                    else:
                        correct_grad = activations[-l - 1][k] * (sp[j] / activations[-l][j]) * \
                                        sum(
                                            self.weights[-l + 1].transpose()[j, k_prime] * nabla_w[-l + 1][k_prime, j]
                                            for k_prime in range(nabla_w[-l + 1].shape[0])
                                        )

                        experimental_grad = sum(
                                            self.weights[-l + 1].transpose()[j, k_prime] * nabla_w[-l + 1][k_prime, j]
                                            for k_prime in range(nabla_w[-l + 1].shape[0])
                                        ) / self.weights[-l][j, k]

                        #nabla_w[-l][j, k] = correct_grad

                        error = None
                        try:
                            error = (correct_grad[0] - experimental_grad) / correct_grad[0]
                            sum_percent_error, count = self.weight_grad_error[-l][j, k]
                            self.weight_grad_error[-l][j, k] = sum_percent_error + error, count + 1
                        except FloatingPointError:
                            pass
                        # print(f"{round(error * 100, 3)}%")

                        # This quantity is the term which we assume is negligible in our approximation
                        # Goal: Get this to always be below 0.01
                        try:
                            expected_inaccuracy = (zs[-l][j, 0] - activations[-l-1][k, 0] * self.weights[-l][j, k]) / activations[-l - 1][k, 0]

                            if abs(expected_inaccuracy) > 0.25 or error is None:
                                nabla_w[-l][j, k] = correct_grad
                            else:
                                nabla_w[-l][j, k] = experimental_grad
                                if abs(error) < 0.01:
                                    self.correct_weight_calculations += 1
                                self.total_weight_calculations += 1
                        except FloatingPointError:
                            nabla_w[-l][j, k] = correct_grad

            for j in range(len(nabla_b[-l])):
                nabla_b[-l][j,0] = sum(
                    nabla_w[-l][j, k] / activations[-l - 1][k,0] for k in range(len(nabla_w[-l][j])) if abs(activations[-l - 1][k,0]) > 0.00001
                ) / len(nabla_w[-l][j])

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # Note: Both elements of the tuples in test_data are numpy arrays of shape (1,1)
        test_results = [(self.get_output(x)[0, 0], y[0, 0]) for (x, y) in test_data]

        loss = sum((y - x) ** 2 for (x, y) in test_results)
        return loss

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


class SparselyConnectedNetwork(Network):
    def __init__(self, sizes, max_inputs_per_node):
        super().__init__(sizes)
        self.weight_mask = list()

        for i, weight_mat in enumerate(self.weights):
            mask = np.zeros(weight_mat.shape)

            # # Randomly generates the mask
            # # Constrains by the number of outputs per node
            # # for col in range(weight_mat.shape[1]):
            # #     if weight_mat.shape[0] < max_inputs_per_node:
            # #         weights_to_keep = list(range(weight_mat.shape[0]))
            # #     else:
            # #         weights_to_keep = random.sample(range(weight_mat.shape[0]), max_inputs_per_node)
            # #     for weight_index in weights_to_keep:
            # #         mask[weight_index][col] = 1
            #
            # # Constrains by the number of inputs per node
            # for row in range(weight_mat.shape[0]):
            #     if weight_mat.shape[1] < max_inputs_per_node:
            #         weights_to_keep = list(range(weight_mat.shape[1]))
            #     else:
            #         weights_to_keep = random.sample(range(weight_mat.shape[1]), max_inputs_per_node)
            #     for weight_index in weights_to_keep:
            #         mask[row][weight_index] = 1

            # Generates the mask deterministically
            # Requirements:
            #       1. Each neuron receives input from no more than max_inputs_per_node neurons
            #           - In matrix language, no more than three indices of each row in the mask are 1's
            # `      2. Each neuron outputs to at least one neuron
            #           - In matrix language, at least one index of each column in the mask is a 1
            col_index = 0
            for row in range(weight_mat.shape[0]):
                for _ in range(max_inputs_per_node):
                    col_index = (col_index + 1) % weight_mat.shape[1]
                    mask[row][col_index] = 1

            if i == len(self.weights) - 1:
                # The output layer retains incoming weights, as they will not be a problem in error calculation
                self.weight_mask.append(np.ones(mask.shape))
            else:
                self.weights[i] = np.multiply(self.weights[i], mask)
                self.weight_mask.append(mask)

    def apply_weight_mask(self):
        for i, weight in enumerate(self.weights):
            self.weights[i] = np.multiply(weight, self.weight_mask[i])

    def update_mini_batch(self, mini_batch, eta):
        super().update_mini_batch(mini_batch, eta)
        self.apply_weight_mask()

    def __str__(self):
        return f"SparseNet{self.sizes}"

    def visualize(self):
        self.apply_weight_mask()
        super().visualize()


if __name__ == '__main__':
    np.seterr(all='warn')

    training_data, validation_data, test_data = \
        chessboard_loader.load_data_wrapper(board_size=BOARD_SIZE, n=NUM_SQUARES_PER_ROW)

    net = Network([2, 20, 20, 20, 1])
    net.SGD(training_data, 1000000, 60, 1, te_data=test_data)

    #net = SparselyConnectedNetwork([2, 1000, 1], 2)
    #net.visualize()
    #net.SGD(training_data, 10000000, 1, 0.5, te_data=test_data)

    print(net.dir_name)

    img_arr = np.ndarray((BOARD_SIZE, BOARD_SIZE))
    for inp, exp_result in training_data:
        x = int(inp[0, 0])
        y = int(inp[1, 0])
        img_arr[x, y] = net.feedforward(inp)[0, 0] * 255

    img = Image.fromarray(np.array(img_arr))
    Image.Image.show(img)
