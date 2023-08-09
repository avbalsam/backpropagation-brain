"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import gzip

from keras.datasets import mnist

import numpy as np


def load_data(board_size: int, n: int):
    """
    Load a chessboard dataset, which maps each possible pixel coordinate to
    its expected color.

    :param board_size: Number of pixels on each row of the board
    :param n: Number of squares on each row of the board
    """

    x_train = [(x, y) for x in range(board_size) for y in range(board_size)]
    y_train = [1 if (x // (board_size / n) % 2) == (y // (board_size / n) % 2) else 0 for (x, y) in x_train]
    x_train = [vectorized_coord(x) for x in x_train]
    y_train = [vectorized_result(y) for y in y_train]

    training_data = (x_train, y_train)
    validation_data = (x_train, y_train)
    test_data = (x_train, y_train)
    return training_data, validation_data, test_data


def load_data_wrapper(board_size: int, n: int):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(board_size, n)
    training_data = list(zip(x_train, y_train))
    validation_data = list(zip(x_val, y_val))
    test_data = list(zip(x_test, y_test))
    return training_data, validation_data, test_data


def vectorized_result(j):
    """Return a 2-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a color (black = 0, white = 1) to
    a vector for classification."""
    e = np.zeros((1, 1))
    e[0] = j
    return e


def vectorized_coord(coord):
    """Return a 2x1 numpy array representing a 2-dimensional coordinate vector."""
    coord_arr = np.ndarray((2, 1))
    (x, y) = coord
    coord_arr[0, 0] = x
    coord_arr[1, 0] = y
    return coord_arr
