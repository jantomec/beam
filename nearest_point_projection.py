import numpy as np
import interpolation as intp


def _nearest_point_projection_(
    interpolation: str,
    X: np.ndarray,
    P: np.ndarray,
    s0: float,
    TOLER: float = 1e-8,
    MAXITER: int = 10
) -> np.ndarray:

    n_nodes = len(X[0])
    u = np.zeros((4))
    u[0] = s0
    R = np.ones((4))
    i = 0
    while np.linalg.norm(R) > TOLER and i < MAXITER:
        K = np.zeros((4,4))
        if interpolation == "Lagrange polynoms":
            R[:3] = P - (
                (X @ intp.lagrange_poly(n_nodes-1, [u[0]])).flatten() +
                u[1:]
            )
            R[3] = 0 - (
                (X @ intp.lagrange_poly_d(n_nodes-1, [u[0]])).flatten()
            ).dot(u[1:])
            K[:3,0] = (
                X @ intp.lagrange_poly_d(n_nodes-1, [u[0]])
            ).flatten()
            K[3,0] = (
                (X @ intp.lagrange_poly_dd(n_nodes-1, [u[0]])).flatten()
            ).dot(u[1:])
            K[3,1:] = (
                X @ intp.lagrange_poly_d(n_nodes-1, [u[0]])
            ).flatten()
        K[:3,1:] = np.identity(3)
        u += np.linalg.solve(K, R)
        i+= 1
    return u

def nearest_point_projection(
    interpolation: str,
    X: np.ndarray,
    P: np.ndarray,
    TOLER: float = 1e-8,
    MAXITER: int = 10
) -> np.ndarray:
    
    l = _nearest_point_projection_(
        interpolation, X, P, -1, TOLER, MAXITER
    )
    r = _nearest_point_projection_(
        interpolation, X, P, 1, TOLER, MAXITER
    )
    if not (-1 <= l[0] <= 1) and (-1 <= r[0] <= 1):
            return r
    elif not (-1 <= r[0] <= 1) and (-1 <= l[0] <= 1):
            return l
    else:
        i = np.argmin((
            np.linalg.norm(l[1:]),
            np.linalg.norm(r[1:])
        ))
        return (l, r)[i]
