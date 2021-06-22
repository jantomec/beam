import numpy as np
from beam import elements as el


def assemble_tangent(
    tangent: np.ndarray,
    elements: list,
    global_coordinates: np.ndarray,
    global_displacements: np.ndarray,
    c: list,
    dt: float,
    beta: float,
    gamma: float
):
    """
    Assembly of a tangent matrix.

    Remarks
      - Array imaging is exploited. Input variable tangent is a ndarray
        and as such able to be manipulated within the function. No
        return is necessary.
    
    Args:
        tangent (np.ndarray): tangent matrix to be manipulated
        elements (list): list of all elements in current iteration
        c (list): list of newmark-beta matrix multipliers
        global_coordinates (np.ndarray): initial global coordinates
        global_displacements (np.ndarray): current global displacements
        dt (float): current time step
        beta (float): newmark-beta parameter
        gamma (float): newmark-beta parameter
    """
    for e in range(len(elements)):
        if type(elements[e]) == el.SimoBeam:
            S = c[0] * elements[e].stiffness_matrix(
                global_coordinates[:,elements[e].nodes] +
                global_displacements[:,elements[e].nodes]
            )
            if c[2] != 0:
                S += c[2] * elements[e].mass_matrix(
                    dt, beta, gamma
                )
            A = elements[e].assemb
            tangent += A @ S @ A.T