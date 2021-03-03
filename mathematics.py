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

    
def rotvec_to_quat(rv):
    angle = np.linalg.norm(rv)
    a = rv
    if angle != 0.0:
        a = a / angle
    q = np.zeros(shape=(4))
    q[0] = a[0] * np.sin(angle/2.0)
    q[1] = a[1] * np.sin(angle/2.0)
    q[2] = a[2] * np.sin(angle/2.0)
    q[3] = np.cos(angle/2.0)
    return q

def quat_to_rotvec(q):
    n = np.linalg.norm(q[:3])
    rv = np.zeros(shape=(3))
    if n != 0.0:
        angle = 2.0 * np.arctan2(n, q[3])
        if angle > 1.8 * np.pi:
            angle = angle - 2.0*np.pi
        if angle < -1.8 * np.pi:
            angle = angle + 2.0*np.pi
        rv = angle * q[:3] / n
    return rv

def spurrier_quat_extraction(R):
    """
     Extraction of quat from rotation matrix R.
    """
    tr = R[0,0] + R[1,1] + R[2,2]
  
    M = max(tr, R[0,0], R[1,1], R[2,2])
    q = np.zeros(shape=(4))

    if M == tr:
        q[3] = 0.5 * np.sqrt(1.0 + tr)
        q[0] = 0.25 *(R[2,1] - R[1,2]) / q[3]
        q[1] = 0.25 *(R[0,2] - R[2,0]) / q[3]
        q[2] = 0.25 *(R[1,0] - R[0,1]) / q[3]
    elif M == R[0,0]:
        q[0] = np.sqrt(0.5 * R[0,0] + 0.25*(1.0 - tr))
        q[3] = 0.25 *(R[2,1] - R[1,2]) / q[0]
        q[1] = 0.25 *(R[1,0] + R[0,1]) / q[0]
        q[2] = 0.25 *(R[2,0] + R[0,2]) / q[0]
    elif M == R[1,1]:
        q[1] = np.sqrt(0.5 * R[1,1] + 0.25*(1.0 - tr))
        q[3] = 0.25 *(R[0,2] - R[2,0]) / q[1]
        q[2] = 0.25 *(R[2,1] + R[1,2]) / q[1]
        q[0] = 0.25 *(R[0,1] + R[1,0]) / q[1]
    elif M == R[2,2]:
        q[2] = np.sqrt(0.5 * R[2,2] + 0.25*(1.0 - tr))
        q[3] = 0.25 *(R[1,0] - R[0,1]) / q[2]
        q[0] = 0.25 *(R[0,2] + R[2,0]) / q[2]
        q[1] = 0.25 *(R[1,2] + R[2,1]) / q[2]

    return normalized(q)

def rotmat_to_quat(R):
    return spurrier_quat_extraction(R)

def quat_to_rotmat(q):
    qc = normalized(q)
    S = skew(qc[:3])
    S2 = S @ S
    R = np.identity(3) + 2.0*qc[3]*S + 2.0*S2
    return R

def hampr(ql: np.ndarray, qr: np.ndarray) -> np.ndarray:
    """
    Hamilton product between two quaternions.
    """
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
