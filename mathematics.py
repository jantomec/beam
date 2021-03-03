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

def skew(r):
    R = np.zeros(shape=(3,3))
    R[0, 1] = - r[2]
    R[0, 2] = r[1]
    R[1, 0] = r[2]
    R[1, 2] = - r[0]
    R[2, 0] = - r[1]
    R[2, 1] = r[0]
    return R

def antiskew(R):
    r = np.zeros(shape=(3))
    r[0] = R[2,1]
    r[1] = R[0,2]
    r[2] = R[1,2]
    return r


    