import numpy as np
import structures as st
import interpolation as intp


class Element:
    def __init__(
        self,
        nodes
    ):
        self.nodes = np.array(nodes)
        self.n_nodes = len(self.nodes)

class SimoBeam(Element):
    def __init__(
        self, 
        nodes,
        coordinates: np.ndarray,
        area: float,
        density: float,
        elastic_modulus: float,
        shear_modulus: float,
        intertia_primary: float,
        intertia_secondary: float,
        inertia_torsion: float,
        shear_coefficient: float = 1
    ):
        # --------------------------------------------------------------
        # nodes
        super().__init__(nodes)
        self.n_dof_per_node = 6

        # --------------------------------------------------------------
        # integration points and interpolation
        lgf = np.polynomial.legendre.leggauss(self.n_nodes)
        lgr = np.polynomial.legendre.leggauss(self.n_nodes - 1)

        self.int_pts = [
            st.BeamIntegrationPoint(
                pointsLocation=lgf[0],
                weights=lgf[1],
                displacement_interpolation=intp.lagrange_poly,
                rotation_interpolation=intp.lagrange_poly,
                n_nodes=self.n_nodes
            ),
            st.BeamIntegrationPoint(
                pointsLocation=lgr[0],
                weights=lgr[1],
                displacement_interpolation=intp.lagrange_poly,
                rotation_interpolation=intp.lagrange_poly,
                n_nodes=self.n_nodes
            )
        ]

        # --------------------------------------------------------------
        # initial element length
        dx = coordinates @ self.int_pts[1].Ndis
        intg = np.zeros(shape=(3))
        for i in range(len(intg)):
            intg[i] = np.dot(dx[i], self.int_pts[1].wgt)
        L = np.linalg.norm(intg)

        # --------------------------------------------------------------
        # element properties
        self.prop = st.BeamElementProperties(
            length=L,
            area=area,
            density=density,
            elastic_modulus=elastic_modulus,
            shear_modulus=shear_modulus,
            intertia_primary=intertia_primary,
            intertia_secondary=intertia_secondary,
            inertia_torsion=inertia_torsion,
            shear_coefficient=shear_coefficient
        )
        

def assembly_matrix(
    nodes: np.ndarray,
    local_dof: np.ndarray,
    n_nodes: int,
    m_dof: int
) -> np.ndarray:
    A = np.zeros((len(local_dof), n_nodes*m_dof), dtype=np.float)
    for i in nodes:
        A[
            local_dof[0]: local_dof[-1] + 1,
            (i*m_dof + local_dof[0]): (i*m_dof + local_dof[-1] + 1)
        ] = np.identity(len(local_dof))
    return A

def constitutive_matrix(
    E: float,
    nu: float,
    A: float,
    I11: float,
    I22: float,
    J: float
) -> np.ndarray:

    G = E / (1+nu)/2
    C = np.zeros((6,6))
    C[0,0] = E * A
    C[1,1] = G * A
    C[2,2] = G * A
    C[3,3] = G * J
    C[4,4] = E * I11
    C[5,5] = E * I22
    return C