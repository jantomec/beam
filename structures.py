import numpy as np
from typing import Callable
import interpolation as intp


class BeamIntegrationPoint:
    """
    A class with all values, stored in integration points for a beam.

    ...

    Attributes
    ----------
    n_pts : int
        number of integration points
    loc : np.ndarray, shape=(n_pts,)
        locations of integration points on the interval [-1, 1]
    wgt : np.ndarray, shape=(n_pts,)
        integration weights
    Ndis : np.ndarray, shape=(n_nodes, n_pts)
        interpolation function matrix for displacement dof
    Nrot : np.ndarray, shape=(n_nodes, n_pts)
        interpolation function matrix for rotation dof
    rot : np.ndarray, shape=(4,n_pts)
        quaternion orientation of the cross-section
    om : np.ndarray, shape=(3,n_pts)
        curvature vector
    w : np.ndarray, shape=(3,n_pts)
        angular velocity vector
    a : np.ndarray, shape=(3,n_pts)
        angular acceleration vector
    q : np.ndarray, shape=(3,n_pts)
        external distributed line load
    f : np.ndarray, shape=(3,n_pts)
        internal distributed forces
    
    Methods
    -------
    
    """
    def __init__(
        self,
        pointsLocation: np.ndarray,
        weights: np.ndarray,
        displacement_interpolation: str,
        rotation_interpolation: str,
        n_nodes
    ):
        """

        """
        self.n_pts = len(pointsLocation)
        self.loc = pointsLocation
        self.wgt = weights
        if displacement_interpolation == "Lagrange polynoms":
            self.Ndis = intp.lagrange_poly(
                n_nodes - 1,
                pointsLocation
            )
            self.dNdis = intp.lagrange_poly_d(
                n_nodes - 1,
                pointsLocation
            )
        if rotation_interpolation == "Lagrange polynoms":
            self.Nrot = intp.lagrange_poly(
                n_nodes - 1,
                pointsLocation
            )
            self.dNrot = intp.lagrange_poly_d(
                n_nodes - 1,
                pointsLocation
            )
        self.rot = None
        self.om = None
        self.w = None
        self.a = None
        self.q = None  # external distributed load
        self.f = np.zeros(shape=(6,self.n_pts))


class BeamElementProperties:
    def __init__(
        self,
        length: float,
        area: float,
        density: float,
        elastic_modulus: float,
        shear_modulus: float,
        intertia_primary: float,
        intertia_secondary: float,
        inertia_torsion: float,
        shear_coefficient: float
    ):
        self.L = length
        self.A = area
        self.rho = density
        self.E = elastic_modulus
        self.G = shear_modulus
        self.I1 = intertia_primary
        self.I2 = intertia_secondary
        self.It = inertia_torsion
        self.ks = shear_coefficient
        self.C = np.zeros(shape=(6,6))
        self.C[0,0] = self.E * self.A
        self.C[1,1] = self.G * self.A
        self.C[2,2] = self.G * self.A
        self.C[3,3] = self.G * self.It
        self.C[4,4] = self.E * self.I1
        self.C[5,5] = self.E * self.I2
        self.I = np.zeros(shape=(3,3))
        self.I[0,0] = self.rho * (self.I1 + self.I2)
        self.I[1,1] = self.rho * self.I1
        self.I[2,2] = self.rho * self.I2
