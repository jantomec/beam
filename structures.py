import numpy as np
from typing import Callable

class Point:
    def __init__(self, x):
        self.x = np.array(x)

class BeamIntegrationPoint:
    def __init__(
        self,
        pointsLocation: np.ndarray,
        weights: np.ndarray,
        displacementInterpolation: Callable[
            [int, np.ndarray], np.ndarray
        ],
        rotationInterpolation: Callable[
            [int, np.ndarray], np.ndarray
        ],
        nNodes: int
    ):
        self.loc = pointsLocation
        self.wgt = weights
        self.Ndis = displacementInterpolation(
            nNodes - 1,
            pointsLocation
        )
        self.Nrot = rotationInterpolation(
            nNodes - 1,
            pointsLocation
        )