import numpy as np
import interpolation as intp
from scipy import optimize
from errors import ConvergenceError


def nearest_point_projection(
    N,
    dN,
    ddN,
    X: np.ndarray,
    P: np.ndarray,
    s0: float = 0.0,
    TOLER: float = 1e-9,
    MAXITER: int = 20
) -> np.ndarray:

    n_nodes = len(X[0])
    u = np.zeros((4))
    u[0] = s0
    R = np.ones((4))
    i = 0
    while np.linalg.norm(R) > TOLER and i < MAXITER:
        K = np.zeros((4,4))
        R[:3] = P - (
            (X @ N([u[0]])).flatten() + u[1:]
        )
        R[3] = 0 - ((X @ dN([u[0]])).flatten()).dot(u[1:])
        K[:3,0] = (X @ dN([u[0]])).flatten()
        K[3,0] = ((X @ ddN([u[0]])).flatten()
        ).dot(u[1:])
        K[3,1:] = (X @ dN([u[0]])).flatten()
        K[:3,1:] = np.identity(3)
        u += np.linalg.solve(K, R)
        i+= 1
    if i == MAXITER:
        raise ConvergenceError("Projection did not converge.")
    return u

# def nearest_point_projection(
#     interpolation: str,
#     X: np.ndarray,
#     P: np.ndarray,
#     TOLER: float = 1e-8,
#     MAXITER: int = 10
# ) -> np.ndarray:
    
#     l = _nearest_point_projection_(
#         interpolation, X, P, -1, TOLER, MAXITER
#     )
#     r = _nearest_point_projection_(
#         interpolation, X, P, 1, TOLER, MAXITER
#     )
#     if not (-1 <= l[0] <= 1) and (-1 <= r[0] <= 1):
#             return r
#     elif not (-1 <= r[0] <= 1) and (-1 <= l[0] <= 1):
#             return l
#     else:
#         i = np.argmin((
#             np.linalg.norm(l[1:]),
#             np.linalg.norm(r[1:])
#         ))
#         return (l, r)[i]

def circular_point_projection(
    interpolation: str,
    X1: np.ndarray,
    X2: np.ndarray,
    s1: float,
    TOLER: float = 1e-8,
    MAXITER: int = 100
) -> np.ndarray:

    n_nodes_1 = len(X1[0])
    n_nodes_2 = len(X2[0])
    x1 = (X1 @ intp.lagrange_polynomial(n_nodes_1-1, [s1])).flatten()
    dx1 = (X1 @ intp.lagrange_polynomial_derivative(n_nodes_1-1, [s1])).flatten()
    x2a = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s1+0.01])).flatten()
    x2b = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s1-0.01])).flatten()
    u = np.zeros((5))
    u[0] = s1
    u[1:3] = (x1 - x2b) / 2
    u[3:5] = (x2a - x1) / 2
    R = np.ones((5))
    i = 0
    while np.linalg.norm(R) > TOLER and i < MAXITER:
        K = np.zeros((5,5))
        if interpolation == "Lagrange polynoms":
            s2 = u[0]
            v1 = u[1:3]
            v2 = u[3:5]
            x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s2])).flatten()
            dx2 = (X2 @ intp.lagrange_polynomial_derivative(n_nodes_2-1, [s2])).flatten()
            ddx2 = (X2 @ intp.lagrange_polynomial_2_derivative(n_nodes_2-1, [s2])).flatten()
            R[:2] = x1 - (x2 + v2 - v1)
            R[2] = 0 - (v2.dot(v2) - v1.dot(v1))
            R[3] = 0 - (v1.dot(dx1))
            R[4] = 0 - (v2.dot(dx2))
            K[0,:2] = dx2
            K[0,2] = 0
            K[0,3] = 0
            K[0,4] = ddx2.dot(v2)
            K[1:3,:2] = -np.identity(2)
            K[1:3,2] = -2*v1
            K[1:3,3] = dx1
            K[1:3,4] = 0
            K[3:5,:2] = np.identity(2)
            K[3:5,2] = 2*v2
            K[3:5,3] = 0
            K[3:5,4] = dx2
        u += np.linalg.solve(K.T, R)
        i+= 1
    if i == MAXITER:
        print("No convergence")
        return False
    return u

def A_(T1, T2, t1, t2):
    return (
        np.cross(t1,t2)*np.dot(T1,T1)
        - 2*np.cross(t1,T1)*np.dot(T1,t2)
        + 2*np.cross(t1,T2)*np.dot(T1,t2)
        - 2*np.cross(t1,t2)*np.dot(T1,T2)
        + 2*np.cross(t1,T1)*np.dot(t2,T2)
        - 2*np.cross(t1,T2)*np.dot(t2,T2)
        + np.cross(t1,t2)*np.dot(T2,T2)
    )
def dA_(T1, T2, t1, t2, dt2):
    return (
        -2*np.cross(t1,T1)*np.dot(T1,dt2)
        + 2*np.cross(t1,T2)*np.dot(T1,dt2)
        + np.cross(t1,dt2)*np.dot(T1,T1)
        - 2*np.cross(t1,dt2)*np.dot(T1,T2)
        + 2*np.cross(t1,T1)*(np.dot(dt2,T2) + np.dot(t2,t2))
        - 2*np.cross(t1,T2)*(np.dot(dt2,T2) + np.dot(t2,t2))
        - 2*np.cross(t1,t2)*np.dot(t2,T2)
        + np.cross(t1,t2)*(np.dot(t2,T2) + np.dot(T2,t2))
        + np.cross(t1,dt2)*np.dot(T2,T2)
    )

def ddA_(T1, T2, t1, t2, dt2, ddt2):
    return (
        -2*np.cross(t1,T1)*np.dot(T1,ddt2)
        + 2*np.cross(t1,T2)*np.dot(T1,ddt2)
        + 2*np.cross(t1,t2)*np.dot(T1,dt2)
        + np.cross(t1,ddt2)*np.dot(T1,T1)
        - 2*np.cross(t1,dt2)*np.dot(T1,t2)
        - 2*np.cross(t1,ddt2)*np.dot(T1,T2)
        + 2*np.cross(t1,T1)*(np.dot(ddt2,T2)
        + 2*np.dot(dt2,t2) + np.dot(t2,dt2))
        - 2*np.cross(t1,T2)*(np.dot(ddt2,T2)
                            + 2*np.dot(dt2,t2)
                            + np.dot(t2,dt2))
        - 4*np.cross(t1,t2)*(np.dot(dt2,T2) + np.dot(t2,t2))
        - 2*np.cross(t1,dt2)*np.dot(t2,T2)
        + np.cross(t1,t2)*(np.dot(dt2,T2) + 2*np.dot(t2,t2) + np.dot(T2,dt2))
        + 2*np.cross(t1,dt2)*(np.dot(t2,T2) + np.dot(T2,t2))
        + np.cross(t1,ddt2)*np.dot(T2,T2)
    )

def B_(T1, T2, t1, t2):
    return (
        -2*t1*np.dot(np.cross(t1,T1),t2)
        - 2*t1*np.dot(np.cross(t1,t2),T2)
    )

def dB_(T1, T2, t1, t2, dt2):
    return (
        -2*t1*np.dot(np.cross(t1,T1),dt2)
        - 2*t1*(np.dot(np.cross(t1,dt2),T2) + np.dot(np.cross(t1,t2),t2))
    )

def ddB_(T1, T2, t1, t2, dt2, ddt2):
    return (
        -2*t1*np.dot(np.cross(t1,T1),ddt2)
        - 2*t1*(np.dot(np.cross(t1,ddt2),T2)
                + 2*np.dot(np.cross(t1,dt2),t2)
                + np.dot(np.cross(t1,t2),dt2))
    )

def _spherical_point_projection(
    interpolation: str,
    X1: np.ndarray,
    X2: np.ndarray,
    s1: float,
    s2_0: float,
    TOLER: float,
    MAXITER: int
) -> tuple:

    n_nodes_1 = len(X1[0])
    n_nodes_2 = len(X2[0])
    x1 = (X1 @ intp.lagrange_polynomial(n_nodes_1-1, [s1])).flatten()
    dx1 = (X1 @ intp.lagrange_polynomial_derivative(n_nodes_1-1, [s1])).flatten()
    s2 = s2_0
    R = 1
    n = 0

    while  abs(R) > TOLER and n < MAXITER:
        K = 0
        if interpolation == "Lagrange polynoms":
            x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s2])).flatten()
            dx2 = (X2 @ intp.lagrange_polynomial_derivative(n_nodes_2-1, [s2])).flatten()
            ddx2 = (X2 @ intp.lagrange_polynomial_2_derivative(n_nodes_2-1, [s2])).flatten()
            dddx2 = (X2 @ intp.lagrange_polynomial_2_derivative(n_nodes_2-1, [s2])).flatten()
            Ai = A_(x1, x2, dx1, dx2)
            dAi = dA_(x1, x2, dx1, dx2, ddx2)
            ddAi = ddA_(x1, x2, dx1, dx2, ddx2, dddx2)
            Bi = B_(x1, x2, dx1, dx2)
            dBi = dB_(x1, x2, dx1, dx2, ddx2)
            ddBi = ddB_(x1, x2, dx1, dx2, ddx2, dddx2)
            dRR = np.zeros(3)
            ddRR = np.ones(3)
            for i in range(3):
                if Bi[i] != 0:
                    dRR[i] = (
                        dAi[i] * Ai[i] / Bi[i] ** 2
                        - Ai[i] ** 2 * dBi[i] / Bi[i] ** 3
                    )
                    ddRR[i] = (
                        ddAi[i] * Ai[i] / Bi[i] ** 2
                        + dAi[i] ** 2 / Bi[i] ** 2
                        -2 * dAi[i] * Ai[i] * dBi[i] / Bi[i] ** 3
                        -2 * Ai[i] * dAi[i] * dBi[i] / Bi[i] ** 3
                        - Ai[i] ** 2 * ddBi[i] / Bi[i] ** 3
                        + 3 * Ai[i] ** 2 * dBi[i] ** 2 / Bi[i] ** 4
                    )
            R = -(dRR).dot(dx1**2)
            K = (ddRR).dot(dx1**2)

        ds = R / K
        s2 += ds
        n+= 1
    if n == MAXITER:
        return False
    x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s2])).flatten()
    dx2 = (X2 @ intp.lagrange_polynomial_derivative(n_nodes_2-1, [s2])).flatten()
    ddx2 = (X2 @ intp.lagrange_polynomial_2_derivative(n_nodes_2-1, [s2])).flatten()
    dddx2 = (X2 @ intp.lagrange_polynomial_2_derivative(n_nodes_2-1, [s2])).flatten()
    Ai = A_(x1, x2, dx1, dx2)
    Bi = B_(x1, x2, dx1, dx2)
    R1 = np.zeros(3)
    for i in range(3):
        if Bi[i] != 0:
            R1[i] = Ai[i] / Bi[i] * dx1[i]
    return (s2, x2, R1)

def spherical_point_projection(
    interpolation: str,
    X1: np.ndarray,
    X2: np.ndarray,
    s1: float,
    n_s2_0: int = 3,
    TOLER: float = 1e-10,
    MAXITER: int = 300
) -> tuple:
    solutions_s2 = np.zeros((n_s2_0))
    solutions_x2 = np.zeros((3,n_s2_0))
    solutions_R1 =  1e+10*np.ones((3,n_s2_0))
    initial_values = np.linspace(-1, 1, n_s2_0 - 2).tolist()
    initial_values += [s1, -s1]

    for (k, s2_0) in enumerate(initial_values):
        sol = _spherical_point_projection(
            "Lagrange polynoms", X1, X2, s1, s2_0=s2_0,
            MAXITER=MAXITER, TOLER=TOLER
        )
        if sol == False:
            continue
        else:
            (s20, x20, R10) = sol
        solutions_s2[k] = s20
        solutions_x2[:,k] = x20
        solutions_R1[:,k] = R10
    sol_i = np.argmin(np.linalg.norm(solutions_R1, axis=0))
    s20 = solutions_s2[sol_i]
    x20 = solutions_x2[:,sol_i]
    R10 = solutions_R1[:,sol_i]
    return (s20, x20, R10)

def spherical_point_projection_2(
    interpolation: str,
    X1: np.ndarray,
    X2: np.ndarray,
    s1: float,
    TOLER: float = 1e-8,
    MAXITER: int = 20
) -> np.ndarray:

    n_nodes_1 = len(X1[0])
    n_nodes_2 = len(X2[0])
    x1 = (X1 @ intp.lagrange_polynomial(n_nodes_1-1, [s1])).flatten()
    dx1 = (X1 @ intp.lagrange_polynomial_derivative(n_nodes_1-1, [s1])).flatten()

    def R1(s):
        x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s])).flatten()
        dx2 = (X2 @ intp.lagrange_polynomial_derivative(n_nodes_2-1, [s])).flatten()
        d = x1 - x2
        v = _triple(dx1, dx2, x1-x2)
        return 1 / v * (
            np.dot(dx2, d) * np.cross(d, dx1)
            + 0.5 * np.dot(d, d) * np.cross(dx1, dx2)
        )

    def f(s):
        Rs = R1(s[0])
        return Rs.dot(Rs)

    sol = optimize.brute(f, [(-1,1)])
    s2 = sol[0]
    x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s2])).flatten()
    R = R1(s2)
    return (s2, x2, R)


def _triple(a, b, c):
    return np.cross(a,b).dot(c)

def spherical_point_projection_3(
    interpolation: str,
    X1: np.ndarray,
    X2: np.ndarray,
    s1: float,
    TOLER: float = 1e-8,
    MAXITER: int = 20
) -> np.ndarray:

    n_nodes_1 = len(X1[0])
    n_nodes_2 = len(X2[0])
    x1 = (X1 @ intp.lagrange_polynomial(n_nodes_1-1, [s1])).flatten()
    dx1 = (X1 @ intp.lagrange_polynomial_derivative(n_nodes_1-1, [s1])).flatten()

    def R1(s):
        x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s])).flatten()
        dx2 = (X2 @ intp.lagrange_polynomial_derivative(n_nodes_2-1, [s])).flatten()
        d = x1 - x2
        v = _triple(dx1, dx2, x1-x2)
        return 1 / v * (
            np.dot(dx2, d) * np.cross(d, dx1)
            + 0.5 * np.dot(d, d) * np.cross(dx1, dx2)
        )
    
    def f(s):
        x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s])).flatten()
        dx2 = (X2 @ intp.lagrange_polynomial_derivative(n_nodes_2-1, [s])).flatten()
        ddx2 = (X2 @ intp.lagrange_polynomial_2_derivative(n_nodes_2-1, [s])).flatten()
        d = x1 - x2
        v = _triple(dx1, dx2, x1-x2)
        R1_ = 1 / v * (
            np.dot(dx2, d) * np.cross(d, dx1) +
            0.5 * np.dot(d, d) * np.cross(dx1, dx2)
        )
        dv = _triple(dx1, ddx2, d)
        dR1_ = 1 / v * (
            1/2 * np.dot(d,d) * np.cross(dx1,ddx2)
            + (np.dot(ddx2,d) - np.dot(dx2,dx2) - dv / v * np.dot(dx2, d)) * np.cross(d, dx1)
            - 1/2 * dv / v * np.dot(d,d) * np.cross(dx1,dx2)
        )
        return np.dot(dR1_, R1_)

    def fprime(s):
        x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s])).flatten()
        dx2 = (X2 @ intp.lagrange_polynomial_derivative(n_nodes_2-1, [s])).flatten()
        ddx2 = (X2 @ intp.lagrange_polynomial_2_derivative(n_nodes_2-1, [s])).flatten()
        d3x2 = (X2 @ intp.lagrange_polynomial_3_derivative(n_nodes_2-1, [s])).flatten()
        d = x1 - x2
        v = _triple(dx1, dx2, x1-x2)
        dv = _triple(dx1, ddx2, d)
        ddv = _triple(dx1, d3x2, d)
        R1_ = 1 / v * (
            np.dot(dx2, d) * np.cross(d, dx1) +
            0.5 * np.dot(d, d) * np.cross(dx1, dx2)
        )
        dR1_ = 1 / v * (
            1/2 * np.dot(d,d) * np.cross(dx1,ddx2)
            + (np.dot(ddx2,d) - np.dot(dx2,dx2) - dv / v * np.dot(dx2, d)) * np.cross(d, dx1)
            - 1/2 * dv / v * np.dot(d,d) * np.cross(dx1,dx2)
        )
        ddR1_ = (
            np.cross(dx1,dx2)*(
                (-(ddv*v) + dv*(2*dv + v))*np.dot(d,d) + 
                2*v**2*np.dot(d,ddx2) +
                2*v**2*np.dot(dx2,dx2)
            ) +
            2*np.cross(d,dx1)*(
                (-(ddv*v) + dv*(2*dv + v))*np.dot(d,dx2) + 
                v*v*np.dot(d,d3x2) - 
                2*v*dv*np.dot(d,ddx2) - 
                3*v**2*np.dot(dx2,ddx2) + 
                2*v*dv*np.dot(dx2,dx2)
            ) + 
            v**2*np.cross(dx1,d3x2)*np.dot(d,d) - 
            2*v*np.cross(dx1,ddx2)*(dv*np.dot(d,d) + v*np.dot(d,dx2))
        )/(2*v**3)
        return np.dot(ddR1_, R1_) + np.dot(dR1_, dR1_)
    # find optimal intial start
    ninit = 20
    s2 = np.linspace(-1, 1, ninit)
    Rs2 = np.zeros_like(s2)
    for i in range(len(s2)):
        Rs2[i] = np.linalg.norm(R1(s2[i]))
    i_min = np.argmin(Rs2)
    try:
        sol = optimize.newton(f, x0=s2[i_min], fprime=fprime, tol=1e-15)
    except RuntimeError:
        print(s2[i_min])
        sol = 0    
    s2 = sol
    x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s2])).flatten()
    R = R1(s2)
    return (s2, x2, R)

def spherical_point_projection_test(
    interpolation: str,
    X1: np.ndarray,
    X2: np.ndarray,
    s1: float,
    TOLER: float = 1e-8,
    MAXITER: int = 20
) -> np.ndarray:

    n_nodes_1 = len(X1[0])
    n_nodes_2 = len(X2[0])
    x1 = (X1 @ intp.lagrange_polynomial(n_nodes_1-1, [s1])).flatten()
    dx1 = (X1 @ intp.lagrange_polynomial_derivative(n_nodes_1-1, [s1])).flatten()

    def R1(s):
        x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s])).flatten()
        dx2 = (X2 @ intp.lagrange_polynomial_derivative(n_nodes_2-1, [s])).flatten()
        d = x1 - x2
        v = _triple(dx1, dx2, x1-x2)
        return 1 / v * (
            np.dot(dx2, d) * np.cross(d, dx1)
            + 0.5 * np.dot(d, d) * np.cross(dx1, dx2)
        )
    def f(s):
        x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s])).flatten()
        dx2 = (X2 @ intp.lagrange_polynomial_derivative(n_nodes_2-1, [s])).flatten()
        ddx2 = (X2 @ intp.lagrange_polynomial_2_derivative(n_nodes_2-1, [s])).flatten()
        d = x1 - x2
        v = _triple(dx1, dx2, x1-x2)
        R1_ = 1 / v * (
            np.dot(dx2, d) * np.cross(d, dx1) +
            0.5 * np.dot(d, d) * np.cross(dx1, dx2)
        )
        dv = _triple(dx1, ddx2, d)
        dR1_ = 1 / v * (
            1/2 * np.dot(d,d) * np.cross(dx1,ddx2)
            + (np.dot(ddx2,d) - np.dot(dx2,dx2) - dv / v * np.dot(dx2, d)) * np.cross(d, dx1)
            - 1/2 * dv / v * np.dot(d,d) * np.cross(dx1,dx2)
        )
        return np.dot(dR1_, R1_)
    
    def fprime(s):
        x2 = (X2 @ intp.lagrange_polynomial(n_nodes_2-1, [s])).flatten()
        dx2 = (X2 @ intp.lagrange_polynomial_derivative(n_nodes_2-1, [s])).flatten()
        ddx2 = (X2 @ intp.lagrange_polynomial_2_derivative(n_nodes_2-1, [s])).flatten()
        d3x2 = (X2 @ intp.lagrange_polynomial_3_derivative(n_nodes_2-1, [s])).flatten()
        d = x1 - x2
        v = _triple(dx1, dx2, x1-x2)
        dv = _triple(dx1, ddx2, d)
        ddv = _triple(dx1, d3x2, d)
        R1_ = 1 / v * (
            np.dot(dx2, d) * np.cross(d, dx1) +
            0.5 * np.dot(d, d) * np.cross(dx1, dx2)
        )
        dR1_ = 1 / v * (
            1/2 * np.dot(d,d) * np.cross(dx1,ddx2)
            + (np.dot(ddx2,d) - np.dot(dx2,dx2) - dv / v * np.dot(dx2, d)) * np.cross(d, dx1)
            - 1/2 * dv / v * np.dot(d,d) * np.cross(dx1,dx2)
        )
        ddR1_ = (
            np.cross(dx1,dx2)*(
                (-(ddv*v) + dv*(2*dv + v))*np.dot(d,d) + 
                2*v**2*np.dot(d,ddx2) +
                2*v**2*np.dot(dx2,dx2)
            ) +
            2*np.cross(d,dx1)*(
                (-(ddv*v) + dv*(2*dv + v))*np.dot(d,dx2) + 
                v*v*np.dot(d,d3x2) - 
                2*v*dv*np.dot(d,ddx2) - 
                3*v**2*np.dot(dx2,ddx2) + 
                2*v*dv*np.dot(dx2,dx2)
            ) + 
            v**2*np.cross(dx1,d3x2)*np.dot(d,d) - 
            2*v*np.cross(dx1,ddx2)*(dv*np.dot(d,d) + v*np.dot(d,dx2))
        )/(2*v**3)
        return np.dot(ddR1_, R1_) + np.dot(dR1_, dR1_)

    s2array = np.linspace(-1,1,200)
    R1array = np.zeros_like(s2array)
    dR1array = np.zeros_like(s2array)
    ddR1array = np.zeros_like(s2array)
    for i in range(s2array.shape[0]):
        R1array[i] = np.linalg.norm(R1(s2array[i]))
        dR1array[i] = f(s2array[i])
        ddR1array[i] = fprime(s2array[i])
    return (s2array, R1array, dR1array, ddR1array)
