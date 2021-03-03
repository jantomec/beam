import numpy as np


def _lagrange_poly_(
    root: int,
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    roots = np.zeros(shape=(degree+1))
    vals = np.ones(shape=(len(eval_pts)))
    for m in range(degree+1):
        roots[m] = 2*m / degree - 1
    for m in range(degree+1):
        if root != m:
            vals *= (eval_pts - roots[m]) / (roots[root] - roots[m])
    return vals

def lagrange_poly(
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    N = np.zeros(shape=(degree+1, len(eval_pts)))
    for j in range(degree+1):
        N[j] = _lagrange_poly_(
            root=j,
            degree=degree,
            eval_pts=eval_pts
        )
    return N

def _lagrange_poly_d_(
    root: int,
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    roots = np.zeros(shape=(degree+1))
    vals = np.zeros(shape=(len(eval_pts)))
    for m in range(degree+1):
        roots[m] = 2*m / degree - 1
    for i in range(degree+1):
        if i != root:
            mvals = np.ones(shape=(len(eval_pts))) 
            for m in range(degree+1):
                if root != m and i != m:
                    mvals += (eval_pts - roots[m]) / (roots[root] - roots[m])
            vals += 1 / (roots[root] - roots[i]) * mvals
    return vals

def lagrange_poly_d(
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    dN = np.zeros(shape=(degree+1, len(eval_pts)))
    for j in range(degree+1):
        dN[j] = _lagrange_poly_d_(
            root=j,
            degree=degree,
            eval_pts=eval_pts
        )
    return dN
