"""
chessboard_random_walk.py
~~~~~~~~~~~~~~~~

Implementation of random walk simulation on chessboard. A mouse, walking randomly across a chessboard,
predicts, at each timestep, the color of its current position. Velocity is modulated by the error,
such that for high error values the velocity of the walk is slower.

Author: Avi Balsam
"""

import itertools
import math
import time
import os

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from seaborn import relplot

# Resolution of test data, and resolution of chessboard representations.
RESOLUTION = (30, 30)

# Number of squares on each row of the chessboard
NUM_SQUARES_PER_ROW = 3

# Changing this variable will change whether the velocity is dependent on error.
ERROR_DEPENDENT_VELOCITY = True


def multilayered_graph(*subset_sizes, weights):
    """Helper function used to visualize the network."""
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
                G.add_edge(layers[layer][col], layers[layer + 1][row], weight=weight)
    return G


def get_expected_output(position, num_squares_per_row):
    """
    Given a 2-d input position where each coordinate is in the range [0, 1], returns the expected output.

    :param position: 2-d coordinates of the mouse
    :param num_squares_per_row: Number of squares in each row of the chessboard.
    """
    x, y = position
    return int(int((x // (1 / num_squares_per_row)) % 2) == int((y // (1 / num_squares_per_row)) % 2))


def generate_test_data(resolution, num_squares_per_row):
    """
    Generates test data for random walk network

    :param resolution: 2d tuple (rows x columns) representing resolution of test image
    :return: Array which maps each pixel to its expected output
    """
    te_data = []
    for row in range(resolution[0]):
        for col in range(resolution[1]):
            x = row / resolution[0]
            y = col / resolution[1]
            te_data.append(
                (
                    np.array(([x], [y])),
                    get_expected_output((x, y), num_squares_per_row)
                )
            )
    return te_data


def generate_training_data(batch_size, base_velocity, num_squares_per_row):
    """
    Returns a function which, when given the initial position of the mouse and the current error vector,
    returns the next batch of values for the position of the mouse, including expected output.

    :param batch_size: Size of each batch
    :param base_velocity: Base velocity of the mouse
    :param num_squares_per_row: Number of squares in each row of the chessboard. Used to compute expected outputs for each square.
    """

    def get_batch(current_position, error):
        """
        Given the current position of the mouse and the error, returns the next batch of the random walk.
        """
        movements = np.random.default_rng().normal(loc=0.0, scale=1.0, size=(batch_size, 2))
        # Divide base velocity by norm of error vector
        if ERROR_DEPENDENT_VELOCITY:
            velocity = base_velocity / math.log(1 + error)
        else:
            velocity = base_velocity
        batch = []
        for movement in movements:
            x, y = current_position
            delta_x, delta_y = movement
            x = (x + delta_x * velocity) % 1
            y = (y + delta_y * velocity) % 1
            current_position = x, y

            batch.append(
                (np.array(current_position), np.array(get_expected_output(current_position, num_squares_per_row)))
            )
        return current_position, batch

    return get_batch


class Network(object):
    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def __init__(self, sizes):
        """
        Given a list of sizes of layers, initializes a network with the specified dimension.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes

        # You can substitute these for any other activation function
        self.activation_func = self.sigmoid
        self.activation_func_deriv = self.sigmoid_prime

        self.dir_name = f"{DIR_NAME}_{str(self)}"

        # We only need biases for the later layers
        self.biases = [np.random.randn(y, 1) + 1 for y in sizes[1:]]

        # Each weight is a matrix with rows = output layer and columns = input layer
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # The length of the error queue is the time delay tau of the network,
        # since errors take tau timesteps to pass through the queue
        self.error_queue = [np.zeros(self.biases[-1].shape) for _ in range(ERROR_TIME_DELAY)]

    def __str__(self):
        return f"Net{self.sizes}"

    def visualize(self):
        """
        Visualize the neurons and connections in the network, and show an image.
        """
        subset_sizes = self.sizes

        G = multilayered_graph(*subset_sizes, weights=self.weights)
        color = ["blue" for v, data in G.nodes(data=True)]
        pos = nx.multipartite_layout(G, subset_key="layer")
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, node_color=color, with_labels=False)
        plt.show()

    def get_output(self, a):
        """
        Return the output of the network if a is input.

        :param a: Input activation
        """
        zs, activations = self.feedforward(a)
        return activations[-1]

    def feedforward(self, x):
        """
        Runs a feedforward pass on input x

        :param x: Input to the network
        """
        activation = x
        activations = [x]
        zs = []
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, activation) + b
            zs.append(z)

            if i == self.num_layers - 2:
                activation = self.sigmoid(z)
            else:
                activation = self.activation_func(z)
            activations.append(activation)
        return zs, activations

    def SGD(self, num_batches: int, batch_size: int, base_velocity: float, eta: float,
            te_data: list):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.

        :param num_batches: Number of batches to train the network with
        :param batch_size: Number of data points in each batch
        :param base_velocity: Base velocity of mouse's random walk
        :param eta: Learning rate of the network
        :param te_data: Test data to evaluate performance of the network
        """
        error_history = list()

        current_position = ([0.5], [0.5])

        get_next_batch = generate_training_data(batch_size, base_velocity, NUM_SQUARES_PER_ROW)

        error = self.evaluate(te_data)

        for j in range(num_batches):
            current_position, batch = get_next_batch(current_position, error)

            self.update_mini_batch(batch, eta)

            if j % 5000 == 0:
                error = self.evaluate(te_data)
                error_history.append([j, error])

            if j % 10000 == 0:
                print("Batch {0}: {1}".format(
                    j, error))

            if j % 10000 == 0:
                if not os.path.exists(f"./{self.dir_name}"):
                    os.makedirs(f"./{self.dir_name}/representations")
                df = pd.DataFrame(error_history, columns=["Epoch", "Error"])
                df.to_csv(f'./{self.dir_name}/error_history.csv')
                g = relplot(df, x="Epoch", y="Error", kind="line")
                g.savefig(f"./{self.dir_name}/error_history.png")
                plt.close()

                # Save a 100x100 snapshot of the chessboard
                img_arr = np.ndarray(RESOLUTION)
                for inp, exp_result in te_data:
                    x = int(inp[0, 0] * RESOLUTION[0])
                    y = int(inp[1, 0] * RESOLUTION[1])
                    out = net.get_output(inp)[0, 0]
                    img_arr[x, y] = out

                img = Image.fromarray(np.array(img_arr) * 255)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(f"./{self.dir_name}/representations/chessboard_representation_epoch:{j}.png")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.

        :param mini_batch: List of tuples (x, y), where x is a training example and y is the expected output
        :param eta: Learning rate of the network"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # You can change which backprop algorithm the network uses in this line.
            delta_nabla_b, delta_nabla_w = self.backprop_attention(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Classical backpropagation algorithm, with no time delay.

        :param x: An input to the network
        :param y: Expected output for input x
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs, activations = self.feedforward(x)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                self.sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_func_deriv(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def backprop_attention(self, x, y):
        """Normal backpropagation, but the error signal is delayed by timesteps equal to the length of the error queue.

        :param x: Input to the network
        :param y: Expected output from input x
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs, activations = self.feedforward(x)

        # backward pass
        correct_delta = self.cost_derivative(activations[-1], y) * \
                        self.sigmoid_prime(zs[-1])
        self.error_queue.append(correct_delta)
        delta = self.error_queue.pop(0)
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
        """
        Backpropagation algorithm which propagates weights instead of a global error term.
        The change in weight from each layer can be used to find the change in weight
        for the previous layer, and to find the biases for the previous layer. This algorithm
        is equivalent to backpropagation, but it is significantly slower, since we can't use
        matrix multiplications.

        :param x: Input to the network
        :param y: Expected output from input x
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs, activations = self.feedforward(x)

        # We only do the first step of the backward pass. This step only depends on the activations of the
        # output layer, and is easy for those neurons to compute.
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
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

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the current performance of the network on the entire chessboard."""
        test_results = [(self.get_output(x)[0, 0], y) for (x, y) in test_data]

        loss = sum((y - x) ** 2 for (x, y) in test_results)
        return loss

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives of the cost function
         in terms of the output activations.

         :param output_activations: Activations of the output layer.
         :param y: Expected values of the output activations."""
        return 2 * (output_activations - y)


if __name__ == '__main__':
    # Finds error and final performance for a variety of time delays tau. This code can be used
    # to reproduce the figures in the paper.
    for delay in [5, 10, 15, 20, 25]:
        for iteration in range(25):
            print(f"Starting model for tau={delay}, iteration: {iteration}...")
            start = time.time()
            ERROR_TIME_DELAY = delay

            # Base dir name to use. Each network will add on to this.
            DIR_NAME = f"random_walk/time_delay_tests_error_dependent_iterations/chessboard_error_time_delay:{ERROR_TIME_DELAY}_numsquares:{NUM_SQUARES_PER_ROW}/iteration_{iteration}"

            net = Network([2, 20, 20, 20, 1])
            if os.path.exists(net.dir_name):
                print(f"Model {net.dir_name} already exists. Skipping...")
                continue

            net.SGD(num_batches=500001, batch_size=10, base_velocity=0.1, eta=0.1,
                    te_data=generate_test_data(RESOLUTION, NUM_SQUARES_PER_ROW))
            print(
                f"Finished model for tau={delay}, iteration: {iteration}\nTime elapsed: {time.time() - start}\n\n\n\n")
