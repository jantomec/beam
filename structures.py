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
        nNodes: int
    ):
        self.loc = pointsLocation
        self.wgt = weights
        self.Ndis = displacement_interpolation(
            nNodes - 1,
            pointsLocation
        )
        self.Nrot = rotation_interpolation(
            nNodes - 1,
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
