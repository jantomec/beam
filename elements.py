import numpy as np
import structures as st
import mathematics as mt


class Element:
    def __init__(
        self,
        nodes,
        local_dof,
        n_nodes_in_mesh,
        mesh_dof_per_node
    ):
        self.nodes = np.array(nodes)
        self.n_nodes = len(self.nodes)
        self.dof = np.array(local_dof)
        
        N = mesh_dof_per_node * n_nodes_in_mesh
        n = len(local_dof)
        self.assemb = np.zeros(shape=(N,n))
        for n in self.nodes:
            self.assemb += assembly_matrix(
                node=n,
                local_dof=self.dof,
                n_nodes_in_mesh=n_nodes_in_mesh,
                mesh_dof_per_node=mesh_dof_per_node
            )

class SimoBeam(Element):
    def __init__(
        self, 
        nodes,
        n_nodes_in_mesh: int,
        mesh_dof_per_node: int,
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
        super().__init__(
            nodes,
            [0,1,2,3,4,5],
            n_nodes_in_mesh,
            mesh_dof_per_node
        )

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

    def stiffness_matrix(self, x: np.ndarray) -> np.ndarray:
        dx = x @ self.int_pts[1].Ndis
        K = np.zeros(shape=(12, 12))

        # --------------------------------------------------------------
        # material part
        for g in range(self.int_pts[1].n_pts):
            c = np.zeros(shape=(6,6))
            # $c = \Pi @ C @ \Pi^T = \Pi (\Pi @ C)^T$ because $C^T = C$
            q = self.int_pts[1].rot[:,g]
            c[:3,:3] = mt.rotate2(
                q,
                (mt.rotate2(q, self.prop.C[:3,:3])).T
            )
            c[3:,3:] = mt.rotate2(
                q,
                (mt.rotate2(q, self.prop.C[3:,3:])).T
            )
            for i in range(self.n_nodes):
                Xi_i = Xi_mat(
                    dx[:,g],
                    self.int_pts[1].dNdis[i,g],
                    self.int_pts[1].Nrot[i,g],
                    self.int_pts[1].dNrot[i,g]
                )
                for j in range(self.n_nodes):
                    Xi_j = Xi_mat(
                        dx[:,g],
                        self.int_pts[1].dNdis[j,g],
                        self.int_pts[1].Nrot[j,g],
                        self.int_pts[1].dNrot[j,g]
                    )
                    K[6*i:6*(i+1), 6*j:6*(j+1)] += (
                        self.int_pts[1].wgt[g] *
                        Xi_i @ c @ Xi_j.T
                    )

        # --------------------------------------------------------------
        # geometric part
        for g in range(self.int_pts[1].n_pts):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    G = np.zeros(shape=(6,6))
                    G[:3,3:] = (
                        -mt.skew(self.int_pts[1].f[:3,g]) *
                        self.int_pts[1].dNdis[i,g] *
                        self.int_pts[1].Nrot[j,g]
                    )
                    G[3:,:3] = (
                        mt.skew(self.int_pts[1].f[:3,g]) *
                        self.int_pts[1].dNdis[j,g] *
                        self.int_pts[1].Nrot[i,g]
                    )
                    G[3:,3:] = (
                        -mt.skew(self.int_pts[1].f[3:,g]) *
                        self.int_pts[1].dNrot[i,g] *
                        self.int_pts[1].Nrot[j,g] +
                        mt.skew(dx[:,g]) @ mt.skew(self.int_pts[1].f[:3,g]) *
                        self.int_pts[1].Nrot[i,g] *
                        self.int_pts[1].Nrot[j,g]
                    )
                    K[6*i:6*(i+1), 6*j:6*(j+1)] += (
                        self.int_pts[1].wgt[g] * G
                    )

        return self.prop.L / 2.0 * K 

    def stiffness_residual(self, x: np.ndarray) -> np.ndarray:
        dx = x @ self.int_pts[1].Ndis
        R = np.zeros(shape=(12))
        for g in range(self.int_pts[1].n_pts):
            for i in range(self.n_nodes):
                Xi_i = Xi_mat(
                    dx[:,g],
                    self.int_pts[1].dNdis[i,g],
                    self.int_pts[1].Nrot[i,g],
                    self.int_pts[1].dNrot[i,g]
                )
                R[6*i:6*(i+1)] += (
                    Xi_i @ self.int_pts[1].f[:,g] *
                    self.int_pts[1].wgt[g]
                )
        return self.prop.L / 2.0 * R

    def mass_matrix(self) -> np.ndarray:
        K = np.zeros(shape=(12,12))
        return K

    def mass_residual(self) -> np.ndarray:
        R = np.zeros(shape=(12))
        return R
    
    def follower_matrix(self) -> np.ndarray:
        K = np.zeros(shape=(12,12))
        return K
    
    def follower_residual(self) -> np.ndarray:
        R = np.zeros(shape=(12))
        return R

class MortarContact(Element):
    def __init__(
        self, 
        nodes,
        n_nodes_in_mesh: int,
        mesh_dof_per_node: int
    ):
        # --------------------------------------------------------------
        # nodes
        super().__init__(
            nodes,
            [6],
            n_nodes_in_mesh,
            mesh_dof_per_node
        )


def Xi_mat(
    dx: np.ndarray,
    dNdis: float,
    Nrot: float,
    dNrot: float
) -> np.ndarray:
    S = mt.skew(dx)
    Xi = np.identity(6)
    Xi[:3] *= dNdis
    Xi[3:] *= dNrot
    Xi[3:,:3] = - Nrot * mt.skew(dx)
    return Xi

def assembly_matrix(
    node: int,
    local_dof: np.ndarray,
    n_nodes_in_mesh: int,
    mesh_dof_per_node: int
) -> np.ndarray:
    
    N = mesh_dof_per_node * n_nodes_in_mesh
    n = len(local_dof)
    A = np.zeros(shape=(N,n))
    for j in range(n):
        A[node*mesh_dof_per_node+j,local_dof[j]] = 1

    return A
