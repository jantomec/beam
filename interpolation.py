import numpy as np


def _lagrange_polynomial_(
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

def lagrange_polynomial(
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    N = np.zeros(shape=(degree+1, len(eval_pts)))
    for j in range(degree+1):
        N[j] = _lagrange_polynomial_(
            root=j,
            degree=degree,
            eval_pts=eval_pts
        )
    return N

def _lagrange_polynomial_derivative_(
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
                    mvals *= (eval_pts - roots[m]) / (roots[root] - roots[m])
            vals += 1 / (roots[root] - roots[i]) * mvals
    return vals

def lagrange_polynomial_derivative(
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    dN = np.zeros(shape=(degree+1, len(eval_pts)))
    for j in range(degree+1):
        dN[j] = _lagrange_polynomial_derivative_(
            root=j,
            degree=degree,
            eval_pts=eval_pts
        )
    return dN

def _lagrange_polynomial_2_derivative_(
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
            mvals = np.zeros(shape=(len(eval_pts)))
            for m in range(degree+1):
                if root != m and i != m:
                    lvals = np.ones(shape=(len(eval_pts)))
                    for l in range(degree+1):
                        if root != l and i != l and m != l:
                            lvals *= (
                                (eval_pts - roots[l]) /
                                (roots[root] - roots[l])
                            )
                    mvals += 1 / (roots[root] - roots[m]) * lvals
            vals += mvals / (roots[root] - roots[i])
    return vals

def lagrange_polynomial_2_derivative(
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    ddN = np.zeros(shape=(degree+1, len(eval_pts)))
    for j in range(degree+1):
        ddN[j] = _lagrange_polynomial_2_derivative_(
            root=j,
            degree=degree,
            eval_pts=eval_pts
        )
    return ddN

def _lagrange_polynomial_3_derivative_(
    root: int,
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    j = root
    roots = np.zeros(shape=(degree+1))
    vals = np.zeros(shape=(len(eval_pts)))
    for m in range(degree+1):
        roots[m] = 2*m / degree - 1
    for i in range(degree+1):
        if i != j:
            lvals = np.zeros(shape=(len(eval_pts)))
            for l in range(degree+1):
                if l != j and l != i:
                    nvals = np.zeros(shape=(len(eval_pts)))
                    for n in range(degree+1):
                        if n !=j and n!= i and n!= l:
                            mvals = np.ones(shape=(len(eval_pts)))
                            for m in range(degree+1):
                                if m != j and m != i and m != l and m != n:
                                    mvals *= (
                                        (eval_pts - roots[m]) /
                                        (roots[j] - roots[m])
                                    )
                            nvals += 1 / (roots[j] - roots[n]) * mvals
                    lvals += 1 / (roots[j] - roots[l]) * nvals
            vals += lvals / (roots[j] - roots[i])
    return vals

def lagrange_polynomial_3_derivative(
    degree: int,
    eval_pts: np.ndarray
) -> np.ndarray:
    d3N = np.zeros(shape=(degree+1, len(eval_pts)))
    for j in range(degree+1):
        d3N[j] = _lagrange_polynomial_3_derivative_(
            root=j,
            degree=degree,
            eval_pts=eval_pts
        )
    return d3N
