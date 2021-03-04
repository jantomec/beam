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
        self.assemb = np.zeros(shape=(N,n*self.n_nodes))
        for i in range(self.n_nodes):
            self.assemb[:,n*i:n*(i+1)] += assembly_matrix(
                node=nodes[i],
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

        # Interpolation derivatives need to be corrected for
        #  isoparametric formulation. This is done, when the element
        #  length is computed.

        # --------------------------------------------------------------
        # initial element length
        dx = coordinates @ self.int_pts[1].dNdis
        intg = np.zeros(shape=(3))
        for i in range(len(intg)):
            intg[i] = np.dot(dx[i], self.int_pts[1].wgt)
        L = np.linalg.norm(intg)

        for i in range(len(self.int_pts)):
            self.int_pts[i].dNdis *= 2 / L
            self.int_pts[i].dNrot *= 2 / L

        # --------------------------------------------------------------
        # initial rotation
        for i in range(len(self.int_pts)):
            self.int_pts[i].rot = np.zeros(
                shape=(3,4,self.int_pts[i].n_pts)
            )
            dx = coordinates @ self.int_pts[i].dNdis
            for g in range(self.int_pts[i].n_pts):
                rotmat = np.zeros(shape=(3,3))
                rotmat[:,0] = mt.normalized(dx[:,g])
                rotmat[:,1] = mt.normalized(
                    np.cross(ref_vec, rotmat[:,0])
                )
                rotmat[:,2] = np.cross(
                    rotmat[:,0], rotmat[:,1]
                )
                self.int_pts[i].rot[:,:,g] = mt.rotmat_to_quat(rotmat)

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

    def update(self, x, th_iter, dt, beta, gamma, iter0 = False):
        """
        x : np.ndarray
            rotational iterative updates
        """
        if iter0:
            for g in range(self.int_pts[0].n_pts):
                # Appropriate updating of rotations is crucial
                # qn_inv = mt.conjugate_quat(self.int_pts[0].rot[0,:,g])
                # q_inv = mt.conjugate_quat(self.int_pts[0].rot[2,:,g])
                # ar1 = mt.quat_to_rotvec(
                #     mt.hamp(self.int_pts[0].rot[2,:,g], qn_inv)
                # )
                # arm1 = mt.rotate(q_inv, ar1)
                # Accumulated rotation arm1 is always zero, except
                #  when there is prescribed rotation.
                a_new = (
                    (1 - 0.5/beta) * self.int_pts[0].a[:,g] -
                    1/(dt*beta) * self.int_pts[0].w[:,g]
                    # + 1/(dt**2*beta) * arm1
                )
                self.int_pts[0].w[:,g] += dt * (
                    (1 - gamma) * self.int_pts[0].a[:,g] +
                    gamma * a_new
                )
                self.int_pts[0].a[:,g] = a_new
            for i in range(len(self.int_pts)):
                self.int_pts[i].rot[0] = self.int_pts[i].rot[2]

        else:
            for i in range(len(self.int_pts)):
                self.int_pts[i].rot[1] = self.int_pts[i].rot[2]
            
            th = th_iter @ self.int_pts[0].Nrot
            for g in range(self.int_pts[0].n_pts):
                qn_inv = mt.conjugate_quat(self.int_pts[0].rot[0,:,g])
                q_inv = mt.conjugate_quat(self.int_pts[0].rot[2,:,g])
                ar1 = mt.quat_to_rotvec(
                    mt.hamp(self.int_pts[0].rot[2,:,g], qn_inv)
                )
                arm1 = mt.rotate(q_inv, ar1)
                dq = mt.rotvec_to_quat(th[:,g])
                self.int_pts[0].rot[2,:,g] = mt.hamp(
                    dq, self.int_pts[0].rot[2,:,g]
                )

                ar2 = mt.quat_to_rotvec(
                    mt.hamp(self.int_pts[0].rot[2,:,g], qn_inv)
                )
                q_inv = mt.conjugate_quat(
                    self.int_pts[0].rot[2,:,g]
                )
                arm2 = mt.rotate(q_inv, ar2)

                self.int_pts[0].w[:,g] += (
                    gamma / (dt*beta) * (arm2 - arm1)
                )
                self.int_pts[0].a[:,g] += (
                    1 / (dt**2*beta) * (arm2 - arm1)
                )

            E1 = np.array([1, 0, 0])
            dx = x @ self.int_pts[1].dNdis
            
            th = th_iter @ self.int_pts[1].Nrot
            dth = th_iter @ self.int_pts[1].dNrot
            for g in range(self.int_pts[1].n_pts):
                dq = mt.rotvec_to_quat(th[:,g])
                self.int_pts[1].rot[2,:,g] = mt.hamp(
                    dq, self.int_pts[1].rot[2,:,g]
                )

                thn = np.linalg.norm(th[:,g])
                if thn == 0:
                    self.int_pts[1].om[:,g] += dth[:,g]
                else:
                    self.int_pts[1].om[:,g] = (
                        (1 - np.sin(thn) / thn) *
                        np.dot(th[:,g], dth[:,g]) /
                        thn ** 2 * th[:,g] +
                        np.sin(thn) / thn * dth[:,g] +
                        (1 - np.cos(thn)) / thn ** 2 *
                        np.cross(th[:,g], dth[:,g]) +
                        np.cos(thn) * self.int_pts[1].om[:,g] +
                        (1 - np.cos(thn)) / thn ** 2 *
                        np.dot(th[:,g], self.int_pts[1].om[:,g]) *
                        th[:,g] + np.sin(thn) / thn * np.cross(
                            th[:,g],
                            self.int_pts[1].om[:,g]
                        )
                    )
                Gamma = mt.rotate(
                    mt.conjugate_quat(self.int_pts[1].rot[2,:,g]),
                    dx[:,g]
                ) - E1
                kappa = mt.rotate(
                    mt.conjugate_quat(self.int_pts[1].rot[2,:,g]),
                    self.int_pts[1].om[:,g]
                )
                fn = self.prop.C[:3,:3] @ Gamma
                fm = self.prop.C[3:,3:] @ kappa
                self.int_pts[1].f[:3,g] = mt.rotate(
                    self.int_pts[1].rot[2,:,g], fn
                )
                self.int_pts[1].f[3:,g] = mt.rotate(
                    self.int_pts[1].rot[2,:,g], fm
                )


    def stiffness_matrix(self, x: np.ndarray) -> np.ndarray:
        dx = x @ self.int_pts[1].dNdis
        K = np.zeros(shape=(12, 12))

        # --------------------------------------------------------------
        # material part
        for g in range(self.int_pts[1].n_pts):
            c = np.zeros(shape=(6,6))
            # $c = \Pi @ C @ \Pi^T = \Pi (\Pi @ C)^T$ because $C^T = C$
            q = self.int_pts[1].rot[2,:,g]
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
        dx = x @ self.int_pts[1].dNdis
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

    def mass_matrix(self, dt, beta, gamma) -> np.ndarray:
        K = np.zeros(shape=(12,12))

        for g in range(self.int_pts[0].n_pts):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    m11 = (
                        np.identity(3) * 
                        self.int_pts[0].Ndis[i,g] *
                        self.int_pts[0].Ndis[j,g]
                    )
                    qn_inv = mt.conjugate_quat(
                        self.int_pts[0].rot[0,:,g]
                    )
                    thg = mt.quat_to_rotvec(
                        mt.hamp(
                            self.int_pts[0].rot[2,:,g],
                            qn_inv
                        )
                    )
                    T = mt.simo_dyn_linmap(mt.skew(thg))
                    Irhoa = self.prop.I @ self.int_pts[0].a
                    wIrhow = np.cross(
                        self.int_pts[0].w[:,g],
                        self.prop.I @ self.int_pts[0].w[:,g]
                    )
                    IrhoawIrhow = Irhoa + wIrhow
                    
                    m22p1 = dt**2 * beta * mt.skew(
                        mt.rotate(
                            self.int_pts[0].rot[2,:,g],
                            IrhoawIrhow
                        )
                    )

                    Irhow = dt * gamma * mt.skew(
                        self.prop.I @ self.int_pts[0].w[:,g]
                    )
                    WIrho = dt * gamma * mt.skew(
                        self.int_pts[0].w[:,g]
                    ) @ self.prop.I
                    m22p2p = self.prop.I - Irhow + WIrho
                    m22p2 = mt.rotate2(
                        self.int_pts[0].rot[2,:,g], m22p2p
                    )

                    m22p3 = (
                        mt.rotate2(qn_inv, T) * 
                        self.int_pts[0].Nrot[i,g] *
                        self.int_pts[0].Nrot[j,g]
                    )
                    
                    m22 = (-m22p1 + m22p2) @ m22p3

                    M = np.zeros(shape=(6,6))
                    M[:3,:3] = m11
                    M[3:,3:] = m22

                    K[6*i:6*(i+1), 6*j:6*(j+1)] += (
                        self.int_pts[1].wgt[g] * M
                    )

        return self.prop.L / 2.0 * K 

    def mass_residual(
        self, global_accelerations
    ) -> np.ndarray:
        R = np.zeros(shape=(12))
        accint = global_accelerations @ self.int_pts[0].Ndis
        for g in range(self.int_pts[0].n_pts):
            for i in range(self.n_nodes):
                f = np.zeros(shape=(6))
                f[:3] = (
                    self.prop.A * self.prop.rho *
                    accint[:,g] *
                    self.int_pts[0].Ndis[i,g]
                )
                f[3:] = (
                    mt.rotate(
                        self.int_pts[0].rot[2,:,g],
                        (
                            self.prop.I @ self.int_pts[0].a[:,g] + np.cross(
                                self.int_pts[0].w[:,g],
                                self.prop.I @ self.int_pts[0].w[:,g]
                            )
                        )
                    ) *
                    self.int_pts[0].Nrot[i,g]
                )
                R[6*i:6*(i+1)] += f * self.int_pts[1].wgt[g]
        return self.prop.L / 2.0 * R
    
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
