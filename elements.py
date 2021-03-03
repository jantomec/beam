import numpy as np
import structures as st
import mathematics as mt


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
        ref_vec: np.ndarray,
        coordinates: np.ndarray,
        angular_velocities: np.ndarray = None,
        angular_accelerations: np.ndarray = None,
        distributed_load: np.ndarray = np.zeros(shape=(6)),
        area: float = 1.0,
        density: float = 0.0,
        elastic_modulus: float = 1.0,
        shear_modulus: float = 1.0,
        inertia_primary: float = 1.0,
        inertia_secondary: float = None,
        inertia_torsion: float = None,
        shear_coefficient: float = 1
    ):
        # --------------------------------------------------------------
        # nodes
        super().__init__(nodes)
        self.n_dof_per_node = 6

        # --------------------------------------------------------------
        # defualt values
        if angular_velocities is None:
            angular_velocities = np.zeros(shape=(3,self.n_nodes))
        if angular_accelerations is None:
            angular_accelerations = np.zeros(shape=(3,self.n_nodes))
        if inertia_secondary is None:
            inertia_secondary = inertia_primary
        if inertia_torsion is None:
            inertia_torsion = inertia_primary

        # --------------------------------------------------------------
        # integration points and interpolation
        lgf = np.polynomial.legendre.leggauss(self.n_nodes)
        lgr = np.polynomial.legendre.leggauss(self.n_nodes - 1)

        self.int_pts = [
            st.BeamIntegrationPoint(
                pointsLocation=lgf[0],
                weights=lgf[1],
                displacement_interpolation="Lagrange polynoms",
                rotation_interpolation="Lagrange polynoms",
                n_nodes=self.n_nodes
            ),
            st.BeamIntegrationPoint(
                pointsLocation=lgr[0],
                weights=lgr[1],
                displacement_interpolation="Lagrange polynoms",
                rotation_interpolation="Lagrange polynoms",
                n_nodes=self.n_nodes
            )
        ]

        # --------------------------------------------------------------
        # initial rotation
        for i in range(len(self.int_pts)):
            self.int_pts[i].rot = np.zeros(
                shape=(4,self.int_pts[i].n_pts)
            )
            dx = coordinates @ self.int_pts[i].Ndis
            for g in range(self.int_pts[i].n_pts):
                rotmat = np.zeros(shape=(3,3))
                rotmat[:,0] = mt.normalized(dx[:,g])
                rotmat[:,1] = mt.normalized(
                    np.cross(ref_vec, rotmat[:,0])
                )
                rotmat[:,2] = np.cross(
                    rotmat[:,0], rotmat[:,1]
                )
                self.int_pts[i].rot[:,g] = mt.rotmat_to_quat(rotmat)

        # --------------------------------------------------------------
        # interpolate velocity, acceleration, load
        self.int_pts[0].w = angular_velocities @ self.int_pts[0].Nrot
        self.int_pts[0].a = angular_accelerations @ self.int_pts[0].Nrot
        self.int_pts[1].om = np.zeros(shape=(3,self.int_pts[1].n_pts))
        self.int_pts[1].q = np.tile(
            distributed_load,
            reps=(self.int_pts[1].n_pts,1)
        ).T

        # --------------------------------------------------------------
        # initial element length
        # reduced int pts dx from before is reused here
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
            inertia_primary=inertia_primary,
            inertia_secondary=inertia_secondary,
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
