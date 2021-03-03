import numpy as np
from typing import Callable


class BeamIntegrationPoint:
    def __init__(
        self,
        pointsLocation: np.ndarray,
        weights: np.ndarray,
        displacement_interpolation: Callable[
            [int, np.ndarray], np.ndarray
        ],
        rotation_interpolation: Callable[
            [int, np.ndarray], np.ndarray
        ],
        n_nodes: int
    ):
        self.loc = pointsLocation
        self.wgt = weights
        self.Ndis = displacement_interpolation(
            n_nodes - 1,
            pointsLocation
        )
        self.Nrot = rotation_interpolation(
            n_nodes - 1,
            pointsLocation
        )

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
