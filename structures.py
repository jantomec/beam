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
        self.rotation_interpolation = rotation_interpolation
        self.displacement_interpolation = displacement_interpolation
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
        self.rot = np.empty(shape=(4,self.n_pts))
        self.om = np.empty(shape=(3,self.n_pts))
        self.w = np.empty(shape=(3,self.n_pts))
        self.a = np.empty(shape=(3,self.n_pts))
        self.q = np.empty(shape=(3,self.n_pts))
        self.f = np.zeros(shape=(6,self.n_pts))


class BeamElementProperties:
    def __init__(
        self,
        length: float,
        area: float,
        density: float,
        elastic_modulus: float,
        shear_modulus: float,
        inertia_primary: float,
        inertia_secondary: float,
        inertia_torsion: float,
        shear_coefficient: float
    ):
        self.L = length
        self.A = area
        self.rho = density
        self.E = elastic_modulus
        self.G = shear_modulus
        self.I1 = inertia_primary
        self.I2 = inertia_secondary
        self.It = inertia_torsion
        self.ks = shear_coefficient
        self.C = np.zeros(shape=(6,6))
        self.C[0,0] = self.E * self.A
        self.C[1,1] = self.G * self.A
        self.C[2,2] = self.G * self.A
        self.C[3,3] = self.G * self.It
        self.C[4,4] = self.E * self.I1
        self.C[5,5] = self.E * self.I2
        self.Arho = area * density
        self.I = np.zeros(shape=(3,3))
        self.I[0,0] = self.rho * (self.I1 + self.I2)
        self.I[1,1] = self.rho * self.I1
        self.I[2,2] = self.rho * self.I2


class MortarIntegrationPoint:
    """
    A class with all values, stored in integration points for a mortar
    contact element.

    ...

    Attributes
    ----------
    n_pts : int
        number of integration points
    loc : np.ndarray, shape=(n_pts,)
        locations of integration points on the interval [-1, 1]
    wgt : np.ndarray, shape=(n_pts,)
        integration weights
    Nlag : np.ndarray, shape=(n_nodes, n_pts)
        interpolation function matrix for lagrange dof
    dNlag : np.ndarray, shape=(n_nodes, n_pts)
        interpolation function derivative matrix for lagrange dof
    Ndis : np.ndarray, shape=(n_nodes, n_pts)
        interpolation function matrix for displacement dof
    dNdis : np.ndarray, shape=(n_nodes, n_pts)
        interpolation function derivative matrix for displacement dof
    cmn : np.ndarray(dtype=int), shape=(1,n_pts)
        closest mortar node
    
    Methods
    -------
    
    """
    def __init__(
        self,
        pointsLocation: np.ndarray,
        weights: np.ndarray,
        lagrange_interpolation: str,
        displacement_interpolation: str,
        n_nodes
    ):
        """

        """
        self.n_pts = len(pointsLocation)
        self.loc = pointsLocation
        self.wgt = weights
        self.lagrange_interpolation = lagrange_interpolation
        self.displacement_interpolation = displacement_interpolation
        if lagrange_interpolation == "Lagrange polynoms":
            self.Nlag = intp.lagrange_poly(
                n_nodes - 1,
                pointsLocation
            )
            self.dNlag = intp.lagrange_poly_d(
                n_nodes - 1,
                pointsLocation
            )
        if displacement_interpolation == "Lagrange polynoms":
            self.Ndis = intp.lagrange_poly(
                n_nodes - 1,
                pointsLocation
            )
            self.dNdis = intp.lagrange_poly_d(
                n_nodes - 1,
                pointsLocation
            )
        self.cmn = np.empty(shape=(self.n_pts), dtype=np.int)
        self.partner = []
