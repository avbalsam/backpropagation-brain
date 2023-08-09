"""
chessboard_opportunistic_neurons.py
~~~~~~~~~~~~

This file is a model of a neural network based on the "opportunistic"
model posited by Dr. Oren Forkosh. Each neuron is assumed to know the
weights of its outgoing synaptic connections, and has control over the
weights of its incoming synaptic connections. Neurons attempt to maximize
the impact they have on their outgoing weights -- in other words, to
be as relevant as possible to the functioning of the network as a whole.

"""

#### Libraries
# Standard library
import random

# Third-party libraries
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from seaborn import relplot
import itertools
import matplotlib.pyplot as plt
import networkx as nx

import chessboard_loader

BOARD_SIZE = 4
NUM_SQUARES_PER_ROW = 2


class Neuron:
    def __init__(self, outgoing_connections: list = None, incoming_connections: list = None, activation_function=None,
                 activation_function_deriv=None):
        """
        Initializes a new neuron

        :param outgoing_connections: List of outgoing Synapses from this neuron.
            In general, this should be initialized to None.
        :param incoming_connections: List of incoming Synapses to this neuron.
            In general, this should be initialized to None.
        """
        # Note: We should never have to look at the presynaptic neuron of incoming connections
        # All the information we need should be accessible from within the Synapse class
        if incoming_connections is None:
            self.incoming_connections = list()
        else:
            self.incoming_connections = incoming_connections

        if outgoing_connections is None:
            self.outgoing_connections = list()
        else:
            self.outgoing_connections = outgoing_connections

        if activation_function is None:
            self.activation_function = sigmoid
            self.activation_function_deriv = sigmoid_prime
        else:
            if not callable(activation_function) or not callable(activation_function_deriv):
                raise ValueError("Uncallable activation functions provided to neuron.")
            self.activation_function = activation_function
            self.activation_function_deriv = activation_function_deriv

        self.activation = None

        # The influence (V) of a neuron is the squared sum of the squares of the weights
        # of its outgoing connections i.e. the impact it has on its postsynaptic connections.
        self.influence = None
        self.delta_v = None  # Change in influence from the last influence update

    def __str__(self):
        return f"N({round(self.activation, 3) if self.activation is not None else None})"

    def update_influence(self):
        # We square each of the weights since we want to measure their power,
        # not their sign.
        # Note: this function does not work if the neuron is in the last layer, and has no outgoing connections
        influence = sum(conn.weight ** 2 for conn in self.outgoing_connections)

        if self.influence is None:
            self.delta_v = influence
        else:
            self.delta_v = influence - self.influence
        self.influence = influence

    def update_involvements(self):
        """Updates the involvements of each of this neuron's incoming connections.
        The involvement of a connection is the extent to which the connection
        contributes to the output of this neuron."""
        total_weighted_input = sum(c.weighted_input for c in self.incoming_connections)
        for conn in self.incoming_connections:
            g = self.activation_function
            conn.involvement = 1 / 2 * (1 - g(total_weighted_input - conn.weighted_input) / g(total_weighted_input))
            assert 0 <= conn.involvement <= 1, "Bad involvement value"

    def update_weights(self, lr):
        """Updates the weights of this neuron's incoming connections based on
        the opportunistic neuron plasticity rule."""
        # Update influence, so that delta_v is correct
        self.update_influence()

        total_weighted_input = sum(c.weighted_input for c in self.incoming_connections)
        for conn in self.incoming_connections:
            g = self.activation_function
            g_prime = self.activation_function_deriv

            conn.delta_w = (lr / 4) * conn.presynaptic_neuron.activation * (
                        g_prime(total_weighted_input) / g(total_weighted_input)
            ) * self.delta_v

            conn.weight += conn.delta_w
            # print(conn.delta_w)

    def activate(self):
        """Applies this neuron's activation function to its total activation.
        Only call this function once all neurons from the previous layer have
        propagated their activations to this neuron."""
        self.activation = self.activation_function(self.activation)

    def propagate(self, a=None):
        """Propagates this neuron's activation (or a given activation value,
        if this neuron is on the first layer) to the next layer of neurons."""
        if a is not None:
            if self.incoming_connections:
                # This neuron is not on layer zero
                raise AttributeError("An activation was passed to a neuron which is not in the base layer")

            self.activation = a

        for conn in self.outgoing_connections:
            conn.propagate()


class Synapse:
    def __init__(self, presynaptic_neuron: Neuron, postsynaptic_neuron: Neuron):
        self.presynaptic_neuron = presynaptic_neuron
        self.postsynaptic_neuron = postsynaptic_neuron
        presynaptic_neuron.outgoing_connections.append(self)
        postsynaptic_neuron.incoming_connections.append(self)

        self.weight = random.random()

        # The extent to which this connection contributes to the final output of the postsynaptic neuron.
        self.involvement = None

        # The weighted input is the number that will be passed to the postsynaptic neuron.
        # We save it for easy access to compute involvement
        self.weighted_input = None

        # Change in weight of the network from the last timestep
        self.delta_w = None

    def __str__(self):
        return f"S({round(self.weight, 3) if self.weight is not None else None})"

    def set_weight(self, weight):
        self.weight = weight

    def propagate(self):
        """Feeds the activation of the presynaptic neuron forward through the network,
        to the postsynaptic neuron, thereby propagating the signal."""
        self.weighted_input = self.presynaptic_neuron.activation * self.weight

        if self.postsynaptic_neuron.activation is None:
            self.postsynaptic_neuron.activation = self.weighted_input
        else:
            self.postsynaptic_neuron.activation += self.weighted_input


class OpportunisticNetwork(object):
    def __init__(self, sizes, activation_function=None, activation_function_deriv=None):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.layers = [[Neuron() for _ in range(y)] for y in sizes]

        self.bias_neuron = Neuron()
        self.bias_neuron.activation = 1

        # The error between this network's output layer and the stimuli
        self.error = None
        # The change in this error over timestep
        self.delta_e = None

        for i, _ in enumerate(self.layers):
            if i == 0:
                continue
            for presynaptic in self.layers[i-1]:
                for postsynaptic in self.layers[i]:
                    Synapse(presynaptic, postsynaptic)
                    Synapse(self.bias_neuron, postsynaptic)

    def visualize(self):
        """Visualize the neurons in this network"""
        G = nx.DiGraph()

        for i, layer in enumerate(self.layers):
            for neuron in layer:
                G.add_node(neuron, layer=i, node_size=neuron.activation)
        G.add_node(self.bias_neuron, layer=0, node_size=1)

        for i, _ in enumerate(self.layers):
            if i == 0:
                continue
            for postsynaptic in self.layers[i]:
                for conn in postsynaptic.incoming_connections:
                    G.add_edge(conn.presynaptic_neuron, postsynaptic, weight=round(conn.weight, 3))

        color = ["gold" for _ in G.nodes(data=True)]
        pos = nx.multipartite_layout(G, subset_key="layer")
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, node_color=color, with_labels=True)
        node_activations = nx.get_node_attributes(G, "node_size")
        nx.draw_networkx_nodes(G, pos, nodelist=node_activations.keys(),
                               node_size=(list(i * 5 for i in node_activations.values()))
                               if not(None in node_activations.values())
                               else 1, alpha=0.8)
        weights = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edges(G, pos,
                               edgelist=weights.keys(),
                               width=list(i * 5 for i in weights.values()),
                               alpha=0.8)
        plt.show()

    def feedforward(self, a):
        """Propagates a signal through the network to the output layer.

        :param a: Numpy array of activations to be passed to the first layer of neurons
        """
        if a.shape[0] != len(self.layers[0]):
            raise ValueError("Activation vector must be the same length as first layer of neurons.")

        for i, neuron in enumerate(self.layers[0]):
            neuron.propagate(a[i, 0])
        self.bias_neuron.propagate()

        for i in range(len(self.layers) - 1):
            for neuron in self.layers[i]:
                neuron.propagate()
            for neuron in self.layers[i + 1]:
                neuron.activate()

        result = np.array([[neuron.activation] for neuron in self.layers[-1]])
        return result

    def reset_activations(self):
        """Sets activations of all neurons equal to None."""
        for layer in self.layers:
            for neuron in layer:
                neuron.activation = None

    def train(self, tr_data, epochs, batch_size, lr, te_data):
        error_history = list()

        tr_data = list(tr_data)
        te_data = list(te_data)

        n = len(tr_data)

        for j in range(epochs):
            random.shuffle(tr_data)
            mini_batches = [
                tr_data[k:k + batch_size]
                for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)

            if te_data:
                error = self.evaluate(te_data)
                error_history.append([j, error])
                print("Epoch {0}: {1}".format(
                    j, error))

            if te_data and j % 10 == 0:
                df = pd.DataFrame(error_history, columns=["Epoch", "Error"])
                g = relplot(df, x="Epoch", y="Error", kind="line")
                g.savefig(f"error_history_chessboard_opp_pixsize:{BOARD_SIZE}_numsquares:{NUM_SQUARES_PER_ROW}.png")

                # Save the chessboard as the NN currently represents it
                img_arr = np.ndarray((BOARD_SIZE, BOARD_SIZE))
                for inp, exp_result in training_data:
                    x = int(inp[0, 0])
                    y = int(inp[1, 0])
                    img_arr[x, y] = self.feedforward(inp)[0, 0] * 255

                img = Image.fromarray(np.array(img_arr))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(
                    f"chessboard_representation_opp_epoch:{j}_pixsize:{BOARD_SIZE}_numsquares:{NUM_SQUARES_PER_ROW}.png")

    def evaluate(self, test_data):
        """Returns the total error of the network."""
        # Note: Both elements of the tuples in test_data are numpy arrays of shape (1,1)
        test_results = [(self.feedforward(x)[0, 0], y[0, 0]) for (x, y) in test_data]

        return round(sum((y - x) ** 2 for (x, y) in test_results), 3)

    def update_mini_batch(self, mini_batch, lr):
        for x, y in mini_batch:
            # Reset the activations of all neurons
            self.reset_activations()

            # Do one feedforward pass
            output = self.feedforward(x)

            # After activating all neurons with a feedforward pass, update connections

            # For the output layer, there are no outgoing connections, so the reward is the negative of the error
            # TODO: Allow for more than one neuron in the last layer by making the error into a vector.
            for i, neuron in enumerate(self.layers[-1]):
                error = (output[i, 0] - neuron.activation) ** 2
                if self.error is None:
                    self.delta_e = error
                else:
                    self.delta_e = error - self.error

                reward = -error
                delta_r = -self.delta_e

                # After computing the reward for the neuron in the last layer,
                # we update the weights of the neurons which connect to it based on the synaptic plasticity rule.
                neuron.update_involvements()
                for conn in neuron.incoming_connections:
                    delta_w = (lr * conn.involvement * delta_r) / (2 * conn.weight)
                    conn.weight += delta_w

            for layer in reversed(self.layers[1:-1]):
                self.visualize()
                for neuron in layer:
                    neuron.update_influence()
                    neuron.update_weights(lr)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == '__main__':
    training_data, validation_data, test_data = \
        chessboard_loader.load_data_wrapper(board_size=BOARD_SIZE, n=NUM_SQUARES_PER_ROW)

    # net = Network([2, 90, 1])
    # net.SGD(training_data, 50000, 30, 0.1, te_data=test_data)
    #
    # # test_results = [[(np.argmax(net.feedforward(np.array([[x], [y]]))), ) for x in range(BOARD_SIZE)]
    # #                 for y in range(BOARD_SIZE)]

    # print(net.feedforward(np.array([[0], [1]])))
    net = OpportunisticNetwork([2, 10, 10, 1])
    net.visualize()
    for conn in net.bias_neuron.outgoing_connections:
        print(conn.weight)
    net.train(training_data, 50, 30, 5, te_data=test_data)
    for conn in net.bias_neuron.outgoing_connections:
        print(conn.weight)
    net.visualize()

    # img_arr = np.ndarray((BOARD_SIZE, BOARD_SIZE))
    # for inp, exp_result in training_data:
    #     x = int(inp[0, 0])
    #     y = int(inp[1, 0])
    #     net = OpportunisticNetwork([2, 30, 4], )
    #     network_output = net.feedforward(inp)[0, 0] * 255
    #     print(network_output)
    #     img_arr[x, y] = network_output
    #
    # img = Image.fromarray(np.array(img_arr))
    # Image.Image.show(img)
