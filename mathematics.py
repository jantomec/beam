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

class quat:
    """
    Quaternion is: [qx, qy, qz, qw]
    """
    def __init__(self, q):
        self.val = np.array(q, dtype=np.float)
    def __repr__(self):
        return self.val.__repr__()
    def __mul__(self, other):
        """
        Hamilton product between two quaternions, also operator *.
        """
        h = quat(np.zeros(shape=(4)))
        h.val[0] = (self.val[3]*other.val[0] + self.val[0]*other.val[3] +
                self.val[1]*other.val[2] - self.val[2]*other.val[1])
        h.val[1] = (self.val[3]*other.val[1] - self.val[0]*other.val[2] +
                self.val[1]*other.val[3] + self.val[2]*other.val[0])
        h.val[2] = (self.val[3]*other.val[2] + self.val[0]*other.val[1] +
                self.val[1]*other.val[0] + self.val[2]*other.val[3])
        h.val[3] = (self.val[3]*other.val[3] - self.val[0]*other.val[0] +
                self.val[1]*other.val[1] - self.val[2]*other.val[2])
        return h


