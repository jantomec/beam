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

def skew(r: np.ndarray) -> np.ndarray:
    R = np.zeros(shape=(3,3))
    R[0, 1] = - r[2]
    R[0, 2] = r[1]
    R[1, 0] = r[2]
    R[1, 2] = - r[0]
    R[2, 0] = - r[1]
    R[2, 1] = r[0]
    return R

def antiskew(R: np.ndarray) -> np.ndarray:
    r = np.zeros(shape=(3))
    r[0] = R[2,1]
    r[1] = R[0,2]
    r[2] = R[1,2]
    return r

def expSO3(R: np.ndarray) -> np.ndarray:
    norm_R = np.linalg.norm(R) / np.sqrt(2)
    if norm_R == 0:
        return np.identity(3)

    return (
        np.identity(3) + 
        np.sin(norm_R) * R / norm_R + 
        (1 - np.cos(norm_R)) * R @ R / norm_R**2
    )
