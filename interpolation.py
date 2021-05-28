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
    eval_pts
) -> np.ndarray:
    try:
        N = np.zeros(shape=(degree+1, len(eval_pts)))
        for j in range(degree+1):
            N[j] = _lagrange_polynomial_(
                root=j,
                degree=degree,
                eval_pts=eval_pts
            )
        return N
    except TypeError:
        N = np.zeros(shape=(degree+1))
        for j in range(degree+1):
            N[j] = _lagrange_polynomial_(
                root=j,
                degree=degree,
                eval_pts=[eval_pts]
            )[0]
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
            mvals = np.ones(len(eval_pts))
            for m in range(degree+1):
                if root != m and i != m:
                    mvals *= (eval_pts - roots[m]) / (roots[root] - roots[m])
            vals += 1 / (roots[root] - roots[i]) * mvals
    return vals

def lagrange_polynomial_derivative(
    degree: int,
    eval_pts
) -> np.ndarray:
    try:
        dN = np.zeros(shape=(degree+1, len(eval_pts)))
        for j in range(degree+1):
            dN[j] = _lagrange_polynomial_derivative_(
                root=j,
                degree=degree,
                eval_pts=eval_pts
            )
        return dN
    except TypeError:
        dN = np.zeros(degree+1)
        for j in range(degree+1):
            dN[j] = _lagrange_polynomial_derivative_(
                root=j,
                degree=degree,
                eval_pts=[eval_pts]
            )[0]
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
    eval_pts
) -> np.ndarray:
    try:
        ddN = np.zeros(shape=(degree+1, len(eval_pts)))
        for j in range(degree+1):
            ddN[j] = _lagrange_polynomial_2_derivative_(
                root=j,
                degree=degree,
                eval_pts=eval_pts
            )
        return ddN
    except TypeError:
        ddN = np.zeros(degree+1)
        for j in range(degree+1):
            ddN[j] = _lagrange_polynomial_2_derivative_(
                root=j,
                degree=degree,
                eval_pts=[eval_pts]
            )[0]
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
    try:
        d3N = np.zeros(shape=(degree+1, len(eval_pts)))
        for j in range(degree+1):
            d3N[j] = _lagrange_polynomial_3_derivative_(
                root=j,
                degree=degree,
                eval_pts=eval_pts
            )
        return d3N
    except TypeError:
        d3N = np.zeros(degree+1)
        for j in range(degree+1):
            d3N[j] = _lagrange_polynomial_3_derivative_(
                root=j,
                degree=degree,
                eval_pts=eval_pts[0]
            )[0]
        return d3N

def dual_basis_function(degree, eval_pts):
    n_integration_points = 2 * (degree + 1) - 1
    (sg, wg) = np.polynomial.legendre.leggauss(n_integration_points)
    phi = lagrange_polynomial(degree, sg)
    a = np.zeros((degree+1, degree+1))
    for j in range(degree+1):
        L = np.zeros((degree+1, degree+1))
        R = np.zeros(degree+1)
        for g in range(n_integration_points):
            L += wg[g] * np.outer(phi[:,g], phi[:,g])
            R[j] += wg[g] * phi[j,g]
        a[j] = np.linalg.solve(L, R)
    return a @ lagrange_polynomial(degree, eval_pts)

def bezier_spline(
    nodes: np.ndarray,
    eval_pts: np.ndarray
) -> np.ndarray:
    # requires larger changes in code (variation and linearization of interpolation functions as they depend on position)
    P1 = nodes[:,0]
    P2 = nodes[:,1]
    P3 = nodes[:,2]
    P4 = nodes[:,3]
    B = np.zeros((3,4))
    B[:,0] = 1.0/4 * (P2 - 2*P3 + P4)
    B[:,1] = 1.0/4 * (P1 - P2 - P3 + P4)
    B[:,2] = 1.0/4 * (-2*P1 - P2 + 4*P3 - P4)
    B[:,3] = 1.0/4 * (P1 + P2 + 3*P3 - P4)
    s = np.array([eval_pts**3, eval_pts**2, eval_pts**1, eval_pts**0])
    return B @ s

def hermite_polynomial(
    nodes: np.ndarray,
    eval_pts: np.ndarray
) -> np.ndarray:
    # requires larger changes in code (variation and linearization of interpolation functions as they depend on position)
    return
