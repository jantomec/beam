import numpy as np
from typing import Callable
import interpolation as intp


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
        shear_coefficient: float,
        contact_radius: float
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
        self.cr = contact_radius
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
        displacement_interpolation,
        rotation_interpolation,
        points_location: np.ndarray = None,
        weights: np.ndarray = None
    ):
        self.n_pts = 0 if points_location is None else len(points_location)
        self.loc = points_location
        self.wgt = weights
        
        # pre-computed values for efficiency
        if self.n_pts > 0:
            self.N_displacement = displacement_interpolation[0](self.loc)
            self.dN_displacement = displacement_interpolation[1](self.loc)
            self.N_rotation = rotation_interpolation[0](self.loc)
            self.dN_rotation = rotation_interpolation[1](self.loc)

        self.rot = np.empty(shape=(3,4,self.n_pts))
        self.om = np.empty(shape=(3,self.n_pts))
        self.w = np.empty(shape=(3,self.n_pts))
        self.a = np.empty(shape=(3,self.n_pts))
        self.q = np.empty(shape=(3,self.n_pts))
        self.f = np.zeros(shape=(6,self.n_pts))


class IntegrationPoint:
    """
    A class with all values, stored in integration points for a finite element.

    ...

    Attributes
    ----------
    n_pts : int
        number of integration points
    loc : np.ndarray, shape=(n_pts,)
        locations of integration points on the interval [-1, 1]
        
    """
    def __init__(
        self,
        point_location: np.ndarray = None,
        weight: np.ndarray = None
    ):
        self.loc = point_location
        self.wgt = weight
