import numpy as np


def normalized(a, axis=-1, order=2):
    """
    Return normalized vector.

    :param a: vector
    :param order: normalization order
    :returns: normalized input
    """
    l2 = np.linalg.norm(a, ord=order)
    return a / l2

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

def hamilton_prod(ql: np.ndarray, qr: np.ndarray) -> np.ndarray:
    h = np.zeros(shape=(4))
    h[0] = (ql[3]*qr[0] + ql[0]*qr[3] +
            ql[1]*qr[2] - ql[2]*qr[1])
    h[1] = (ql[3]*qr[1] - ql[0]*qr[2] +
            ql[1]*qr[3] + ql[2]*qr[0])
    h[2] = (ql[3]*qr[2] + ql[0]*qr[1] +
            ql[1]*qr[0] + ql[2]*qr[3])
    h[3] = (ql[3]*qr[3] - ql[0]*qr[0] +
            ql[1]*qr[1] - ql[2]*qr[2])
    return h
