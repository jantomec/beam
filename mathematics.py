import numpy as np


def normalized(a, axis=-1, order=2):
    """
    Return normalized input, either a vector or a matrix.
    Copied from: https://stackoverflow.com/questions/3898572/what-is-the-standard-python-docstring-format

    :param a: vector or matrix
    :param axis: normalize along this axis
    :param order: normalization order
    :returns: normalized input
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)