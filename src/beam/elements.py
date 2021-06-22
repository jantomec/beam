import numpy as np
from beam import structures as struct
from beam import mathematics as math
from beam import projection as proj
from beam import interpolation as intp
from beam import contact
from beam.errors import ConvergenceError, MaterialError


class Element:
    def __init__(
        self,
        nodes,
        local_dof
    ):
        self.nodes = np.array(nodes)
        self.n_nodes = len(self.nodes)
        self.dof = np.array(local_dof)
        
    def construct_assembly_matrix(self, n_nodes_in_mesh):
        n_all = len(self.dof)
        n_loc = np.sum(self.dof)
        N = n_all * n_nodes_in_mesh
        self.assemb = np.zeros(shape=(N,n_loc*self.n_nodes))
        for i in range(self.n_nodes):
            self.assemb[:,n_loc*i:n_loc*(i+1)] += assembly_matrix(
                node=self.nodes[i],
                local_dof=self.dof,
                n_nodes_in_mesh=n_nodes_in_mesh
            )


class SimoBeam(Element):
    def __init__(
        self,
        nodes,
        mesh_dof_per_node: int,
        ref_vec: np.ndarray,
        coordinates: np.ndarray,
        material,
        angular_velocities: np.ndarray = None,
        angular_accelerations: np.ndarray = None,
        distributed_load: np.ndarray = np.zeros(shape=(6)),
        displacement_interpolation: str = "Lagrange polynomials"
    ):
        # --------------------------------------------------------------
        # nodes
        dof = np.zeros(mesh_dof_per_node, dtype=np.bool)
        dof[:6] = True
        super().__init__(nodes, dof)

        # --------------------------------------------------------------
        # defualt values
        if angular_velocities is None:
            angular_velocities = np.zeros(shape=(3,self.n_nodes))
        if angular_accelerations is None:
            angular_accelerations = np.zeros(shape=(3,self.n_nodes))

        # --------------------------------------------------------------
        # displacement interpolation
        if displacement_interpolation == "Lagrange polynomials":
            self.Ndis = [
                lambda x: intp.lagrange_polynomial(self.n_nodes-1, x),
                lambda x: intp.lagrange_polynomial_derivative(self.n_nodes-1, x),
                lambda x: intp.lagrange_polynomial_2_derivative(self.n_nodes-1, x),
                lambda x: intp.lagrange_polynomial_3_derivative(self.n_nodes-1, x)
            ]
        elif displacement_interpolation == "Bezier spline":
            self.Ndis = [
                lambda x: intp.lagrange_polynomial(self.n_nodes-1, x),
                lambda x: intp.lagrange_polynomial_derivative(self.n_nodes-1, x),
                lambda x: intp.lagrange_polynomial_2_derivative(self.n_nodes-1, x),
                lambda x: intp.lagrange_polynomial_3_derivative(self.n_nodes-1, x)
            ]
        elif displacement_interpolation == "Hermite polynomials":
            self.Ndis = [
                lambda x: intp.lagrange_polynomial(self.n_nodes-1, x),
                lambda x: intp.lagrange_polynomial_derivative(self.n_nodes-1, x),
                lambda x: intp.lagrange_polynomial_2_derivative(self.n_nodes-1, x),
                lambda x: intp.lagrange_polynomial_3_derivative(self.n_nodes-1, x)
            ]
        else:
            raise Exception("Unimplemented interpolation type. Possible: Lagrange polynomials (default), Bezier spline, Hermite polynomials")

        # rotation interpolation
        self.Nrot = [
            lambda x: intp.lagrange_polynomial(self.n_nodes-1, x),
            lambda x: intp.lagrange_polynomial_derivative(self.n_nodes-1, x),
            lambda x: intp.lagrange_polynomial_2_derivative(self.n_nodes-1, x),
            lambda x: intp.lagrange_polynomial_3_derivative(self.n_nodes-1, x)
        ]

        # integration points
        lgf = np.polynomial.legendre.leggauss(self.n_nodes)
        lgr = np.polynomial.legendre.leggauss(self.n_nodes - 1)
        self.int_pts = [
            struct.BeamIntegrationPoint(
                displacement_interpolation=self.Ndis,
                rotation_interpolation=self.Nrot,
                points_location=lgf[0],
                weights=lgf[1]
            ),
            struct.BeamIntegrationPoint(
                displacement_interpolation=self.Ndis,
                rotation_interpolation=self.Nrot,
                points_location=lgr[0],
                weights=lgr[1]
            )
        ]
        # Interpolation derivatives need to be corrected for
        #  isoparametric formulation (inverse of jacobian). This is
        #  done, when the element length is computed.

        # --------------------------------------------------------------
        # initial element length
        dxds = coordinates @ self.int_pts[1].dN_displacement
        intg = np.zeros(shape=(3))
        for i in range(len(intg)):
            intg[i] = np.dot(dxds[i], self.int_pts[1].wgt)
        L = np.linalg.norm(intg)
        self.jacobian = L / 2

        # --------------------------------------------------------------
        # initial rotation
        for i in range(len(self.int_pts)):
            self.int_pts[i].rot = np.zeros(
                shape=(3,4,self.int_pts[i].n_pts)
            )
            dx = 1/self.jacobian * coordinates @ self.int_pts[i].dN_displacement
            for g in range(self.int_pts[i].n_pts):
                rotmat = np.zeros(shape=(3,3))
                rotmat[:,0] = math.normalized(dx[:,g])
                rotmat[:,1] = math.normalized(
                    np.cross(ref_vec, rotmat[:,0])
                )
                rotmat[:,2] = np.cross(
                    rotmat[:,0], rotmat[:,1]
                )
                self.int_pts[i].rot[:,:,g] = math.rotmat_to_quat(rotmat)

        # --------------------------------------------------------------
        # interpolate velocity, acceleration, load
        self.int_pts[0].w[2] = angular_velocities @ self.int_pts[0].N_rotation
        self.int_pts[0].a[2] = angular_accelerations @ self.int_pts[0].N_rotation
        self.int_pts[1].om[2] = np.zeros(shape=(3,self.int_pts[1].n_pts))
        self.int_pts[1].q[2] = np.tile(
            distributed_load,
            reps=(self.int_pts[1].n_pts,1)
        ).T

        # --------------------------------------------------------------
        # element properties
        self.prop = struct.BeamElementProperties(L, material)

    def disp_shape_fun(self, int_points_locations):
        return intp.lagrange_polynomial(self.n_nodes-1, int_points_locations)

    def update(self, x, th_iter, dt, beta, gamma, iter0 = False):
        """
        x : np.ndarray
            rotational iterative updates
        """
        if iter0:
            for i in range(len(self.int_pts)):
                self.int_pts[i].rot[0] = self.int_pts[i].rot[2]
            self.int_pts[0].w[0] = self.int_pts[0].w[2]
            self.int_pts[0].a[0] = self.int_pts[0].a[2]
            self.int_pts[1].om[0] = self.int_pts[1].om[2]
            self.int_pts[1].q[0] = self.int_pts[1].q[2]
            self.int_pts[1].f[0] = self.int_pts[1].f[2]
            for g in range(self.int_pts[0].n_pts):
                # Appropriate updating of rotations is crucial
                # qn_inv = math.conjugate_quat(self.int_pts[0].rot[0,:,g])
                # q_inv = math.conjugate_quat(self.int_pts[0].rot[2,:,g])
                # ar1 = math.quat_to_rotvec(
                #     math.hamp(self.int_pts[0].rot[2,:,g], qn_inv)
                # )
                # arm1 = math.rotate(q_inv, ar1)
                # Accumulated rotation arm1 is always zero, except
                #  when there is prescribed rotation.
                a_new = (
                    (1 - 0.5/beta) * self.int_pts[0].a[2,:,g] -
                    1/(dt*beta) * self.int_pts[0].w[2,:,g]
                    # + 1/(dt**2*beta) * arm1
                )
                self.int_pts[0].w[2,:,g] += dt * (
                    (1 - gamma) * self.int_pts[0].a[2,:,g] +
                    gamma * a_new
                )
                self.int_pts[0].a[2,:,g] = a_new
        else:
            for i in range(len(self.int_pts)):
                self.int_pts[i].rot[1] = self.int_pts[i].rot[2]
            
            th = th_iter @ self.int_pts[0].N_rotation
            for g in range(self.int_pts[0].n_pts):
                qn_inv = math.conjugate_quat(self.int_pts[0].rot[0,:,g])
                q_inv = math.conjugate_quat(self.int_pts[0].rot[2,:,g])
                ar1 = math.quat_to_rotvec(
                    math.hamp(self.int_pts[0].rot[2,:,g], qn_inv)
                )
                arm1 = math.rotate(q_inv, ar1)
                dq = math.rotvec_to_quat(th[:,g])
                self.int_pts[0].rot[2,:,g] = math.hamp(
                    dq, self.int_pts[0].rot[2,:,g]
                )

                ar2 = math.quat_to_rotvec(
                    math.hamp(self.int_pts[0].rot[2,:,g], qn_inv)
                )
                q_inv = math.conjugate_quat(
                    self.int_pts[0].rot[2,:,g]
                )
                arm2 = math.rotate(q_inv, ar2)
                iterative_rotation_change = arm2 - arm1
                self.int_pts[0].w[2,:,g] += (
                    gamma / (dt*beta) * iterative_rotation_change
                )
                self.int_pts[0].a[2,:,g] += (
                    1 / (dt**2*beta) * iterative_rotation_change
                )

        E1 = np.array([1, 0, 0])
        dx = 1 / self.jacobian * x @ self.int_pts[1].dN_displacement
        th = th_iter @ self.int_pts[1].N_rotation
        dth = 1 / self.jacobian * th_iter @ self.int_pts[1].dN_rotation
        for g in range(self.int_pts[1].n_pts):
            dq = math.rotvec_to_quat(th[:,g])
            self.int_pts[1].rot[2,:,g] = math.hamp(
                dq, self.int_pts[1].rot[2,:,g]
            )

            thn = np.linalg.norm(th[:,g])
            if thn == 0:
                self.int_pts[1].om[2,:,g] += dth[:,g]
            else:
                self.int_pts[1].om[2,:,g] = (
                    (1 - np.sin(thn) / thn) *
                    np.dot(th[:,g], dth[:,g]) /
                    thn ** 2 * th[:,g] +
                    np.sin(thn) / thn * dth[:,g] +
                    (1 - np.cos(thn)) / thn ** 2 *
                    np.cross(th[:,g], dth[:,g]) +
                    np.cos(thn) * self.int_pts[1].om[2,:,g] +
                    (1 - np.cos(thn)) / thn ** 2 *
                    np.dot(th[:,g], self.int_pts[1].om[2,:,g]) *
                    th[:,g] + np.sin(thn) / thn * np.cross(
                        th[:,g],
                        self.int_pts[1].om[2,:,g]
                    )
                )
            Gamma = math.rotate(
                math.conjugate_quat(self.int_pts[1].rot[2,:,g]),
                dx[:,g]
            ) - E1
            kappa = math.rotate(
                math.conjugate_quat(self.int_pts[1].rot[2,:,g]),
                self.int_pts[1].om[2,:,g]
            )
            
            fn = self.prop.C[:3,:3] @ Gamma
            fm = self.prop.C[3:,3:] @ kappa
            self.int_pts[1].f[2,:3,g] = math.rotate(
                self.int_pts[1].rot[2,:,g], fn
            )
            self.int_pts[1].f[2,3:,g] = math.rotate(
                self.int_pts[1].rot[2,:,g], fm
            )

    def stiffness_matrix(self, x: np.ndarray) -> np.ndarray:
        dx = 1 / self.jacobian * x @ self.int_pts[1].dN_displacement
        K = np.zeros(shape=(6*self.n_nodes, 6*self.n_nodes))
        
        # --------------------------------------------------------------
        # material part
        for g in range(self.int_pts[1].n_pts):
            c = np.zeros(shape=(6,6))
            # $c = \Pi @ C @ \Pi^T = \Pi (\Pi @ C)^T$ because $C^T = C$
            q = self.int_pts[1].rot[2,:,g]
            c[:3,:3] = math.rotate2(
                q,
                (math.rotate2(q, self.prop.C[:3,:3])).T
            )
            c[3:,3:] = math.rotate2(
                q,
                (math.rotate2(q, self.prop.C[3:,3:])).T
            )
            
            for i in range(self.n_nodes):
                Xi_i = Xi_mat(
                    dx[:,g],
                    1 / self.jacobian * self.int_pts[1].dN_displacement[i,g],
                    self.int_pts[1].N_rotation[i,g],
                    1 / self.jacobian * self.int_pts[1].dN_rotation[i,g]
                )
                for j in range(self.n_nodes):
                    Xi_j = Xi_mat(
                        dx[:,g],
                        1 / self.jacobian * self.int_pts[1].dN_displacement[j,g],
                        self.int_pts[1].N_rotation[j,g],
                        1 / self.jacobian * self.int_pts[1].dN_rotation[j,g]
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
                        -math.skew(self.int_pts[1].f[2,:3,g]) *
                        1 / self.jacobian * self.int_pts[1].dN_displacement[i,g] *
                        self.int_pts[1].N_rotation[j,g]
                    )
                    G[3:,:3] = (
                        math.skew(self.int_pts[1].f[2,:3,g]) *
                        1 / self.jacobian * self.int_pts[1].dN_displacement[j,g] *
                        self.int_pts[1].N_rotation[i,g]
                    )
                    G[3:,3:] = (
                        -math.skew(self.int_pts[1].f[2,3:,g]) *
                        1 / self.jacobian * self.int_pts[1].dN_rotation[i,g] *
                        self.int_pts[1].N_rotation[j,g] +
                        math.skew(dx[:,g]) @ math.skew(self.int_pts[1].f[2,:3,g]) *
                        self.int_pts[1].N_rotation[i,g] *
                        self.int_pts[1].N_rotation[j,g]
                    )
                    K[6*i:6*(i+1), 6*j:6*(j+1)] += (
                        self.int_pts[1].wgt[g] * G
                    )

        return self.jacobian * K 

    def stiffness_residual(self, x: np.ndarray) -> np.ndarray:
        dx = 1 / self.jacobian * x @ self.int_pts[1].dN_displacement
        R = np.zeros(shape=(6*self.n_nodes))
        for g in range(self.int_pts[1].n_pts):
            # internal distributed forces
            for i in range(self.n_nodes):
                Xi_i = Xi_mat(
                    dx[:,g],
                    1 / self.jacobian * self.int_pts[1].dN_displacement[i,g],
                    self.int_pts[1].N_rotation[i,g],
                    1 / self.jacobian * self.int_pts[1].dN_rotation[i,g]
                )
                R[6*i:6*(i+1)] += (
                    Xi_i @ self.int_pts[1].f[2,:,g] *
                    self.int_pts[1].wgt[g]
                )
                # external distributed forces
                R[6*i:6*i+3] -= self.int_pts[1].N_displacement[i,g] * self.int_pts[1].q[2,:3,g]
                R[6*i+3:6*(i+1)] -= self.int_pts[1].N_rotation[i,g] * self.int_pts[1].q[2,3:,g]

        return self.jacobian * R

    def mass_matrix(self, dt, beta, gamma) -> np.ndarray:
        K = np.zeros(shape=(6*self.n_nodes,6*self.n_nodes))

        for g in range(self.int_pts[0].n_pts):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    m11 = self.prop.Arho * (
                        np.identity(3) * 
                        self.int_pts[0].N_displacement[i,g] *
                        self.int_pts[0].N_displacement[j,g]
                    )
                    qn_inv = math.conjugate_quat(
                        self.int_pts[0].rot[0,:,g]
                    )
                    thg = math.quat_to_rotvec(
                        math.hamp(
                            self.int_pts[0].rot[2,:,g],
                            qn_inv
                        )
                    )
                    T = math.simo_dyn_linmap(math.skew(thg))
                    Irhoa = self.prop.Irho @ self.int_pts[0].a[2,:,g]
                    wIrhow = np.cross(
                        self.int_pts[0].w[2,:,g],
                        self.prop.Irho @ self.int_pts[0].w[2,:,g]
                    )
                    IrhoawIrhow = Irhoa + wIrhow
                    
                    m22p1 = dt**2 * beta * math.skew(
                        math.rotate(
                            self.int_pts[0].rot[2,:,g],
                            IrhoawIrhow
                        )
                    )

                    Irhow = dt * gamma * math.skew(
                        self.prop.Irho @ self.int_pts[0].w[2,:,g]
                    )
                    WIrho = dt * gamma * math.skew(
                        self.int_pts[0].w[2,:,g]
                    ) @ self.prop.Irho
                    m22p2p = self.prop.Irho - Irhow + WIrho
                    m22p2 = math.rotate2(
                        self.int_pts[0].rot[2,:,g], m22p2p
                    )

                    m22p3 = (
                        math.rotate2(qn_inv, T) * 
                        self.int_pts[0].N_rotation[i,g] *
                        self.int_pts[0].N_rotation[j,g]
                    )
                    
                    m22 = (-m22p1 + m22p2) @ m22p3

                    M = np.zeros(shape=(6,6))
                    M[:3,:3] = m11
                    M[3:,3:] = m22

                    K[6*i:6*(i+1), 6*j:6*(j+1)] += (
                        self.int_pts[0].wgt[g] * M
                    )

        return self.jacobian * K 

    def mass_residual(
        self, global_accelerations
    ) -> np.ndarray:
        R = np.zeros(shape=(6*self.n_nodes))
        accint = global_accelerations @ self.int_pts[0].N_displacement
        for g in range(self.int_pts[0].n_pts):
            for i in range(self.n_nodes):
                f = np.zeros(shape=(6))
                f[:3] = (
                    self.prop.Arho *
                    accint[:,g] *
                    self.int_pts[0].N_displacement[i,g]
                )
                f[3:] = (
                    math.rotate(
                        self.int_pts[0].rot[2,:,g],
                        (
                            self.prop.Irho @ self.int_pts[0].a[2,:,g] + np.cross(
                                self.int_pts[0].w[2,:,g],
                                self.prop.Irho @ self.int_pts[0].w[2,:,g]
                            )
                        )
                    ) *
                    self.int_pts[0].N_rotation[i,g]
                )
                R[6*i:6*(i+1)] += f * self.int_pts[0].wgt[g]
        return self.jacobian * R
    
    def follower_matrix(self) -> np.ndarray:
        K = np.zeros(shape=(6*self.n_nodes,6*self.n_nodes))
        return K
    
    def follower_residual(self) -> np.ndarray:
        R = np.zeros(shape=(6*self.n_nodes))
        return R

    def compute_momentum(self, X, V):
        p = np.zeros(6)
        
        x = X @ self.int_pts[0].N_displacement
        v = V @ self.int_pts[0].N_displacement
        p_linear = self.jacobian * self.prop.Arho * v @ self.int_pts[0].wgt
        
        p_angular = np.zeros(3)
        for g in range(self.int_pts[0].n_pts):
            p_angular += self.jacobian * self.int_pts[0].wgt[g] * (
                np.cross(x[:,g], self.prop.Arho * v[:,g]) +
                math.rotate(self.int_pts[0].rot[2,:,g], self.prop.Irho @ self.int_pts[0].w[2,:,g])
            )

        p[:3] = p_linear
        p[3:6] = p_angular
        return p

    def compute_kinetic_energy(self, V):
        ek = 0.0
        v = V @ self.int_pts[0].N_displacement
        for g in range(self.int_pts[0].n_pts):
            ek += 1/2 * self.jacobian * self.int_pts[0].wgt[g] * (
                self.prop.Arho * v[:,g] @ v[:,g] + 
                self.int_pts[0].w[2,:,g] @ self.prop.Irho @ self.int_pts[0].w[2,:,g]
            )
        return ek

    def compute_potential_energy(self, X):
        ep = 0.0
        dx = 1 / self.jacobian * X @ self.int_pts[1].dN_displacement
        E1 = np.array([1.0, 0.0, 0.0])
        Cn = self.prop.C[:3,:3]
        Cm = self.prop.C[3:,3:]
        for g in range(self.int_pts[1].n_pts):
            Gamma = math.rotate(math.conjugate_quat(self.int_pts[1].rot[2,:,g]), dx[:,g]) - E1
            kappa = math.rotate(math.conjugate_quat(self.int_pts[1].rot[2,:,g]), self.int_pts[1].om[2,:,g])
            ep += 1/2 * self.jacobian * self.int_pts[1].wgt[g] * (
                Gamma @ Cn @ Gamma + kappa @ Cm @ kappa
            )
        return ep


class MortarContact(Element):
    def __init__(
        self,
        parent_element: int,
        n_integration_points: int,
        possible_contact_partners: list,
        dual_basis_functions: bool
    ):
        # --------------------------------------------------------------
        # nodes
        self.parent = parent_element
        self.dof = np.zeros_like(self.parent.dof)
        self.dof[6] = True
        self.possible_contact_partners = possible_contact_partners

        # Lagrange multiplier interpolation
        if dual_basis_functions == True:
            self.Nlam = [
                lambda x: intp.dual_basis_function(self.parent.n_nodes-1, x)
            ]
        else:
            self.Nlam = [
                lambda x: intp.lagrange_polynomial(self.parent.n_nodes-1, x)
            ]
        
        
        lg = np.polynomial.legendre.leggauss(n_integration_points)
        self.int_pts = [struct.IntegrationPoint(
            point_location=lg[0][g],
            weight=lg[1][g]
        ) for g in range(len(lg[0]))]

        # pre-computed values for efficiency
        if len(self.int_pts) > 0:
            self.N_displacement = self.parent.Ndis[0]([self.int_pts[g].loc for g in range(len(self.int_pts))])
            self.dN_displacement = self.parent.Ndis[1]([self.int_pts[g].loc for g in range(len(self.int_pts))])
            self.N_lagrange = self.Nlam[0]([self.int_pts[g].loc for g in range(len(self.int_pts))])

    def find_partner(self, X):
        
        # integration point positions
        x1 = X[:,self.parent.nodes] @ self.N_displacement

        for g in range(len(self.int_pts)):
            
            candidate_elements = self.possible_contact_partners
            projections_all = []
            for candidate in candidate_elements:
                try:
                    projection = proj.nearest_point_projection(
                        candidate.Ndis[0],
                        candidate.Ndis[1],
                        candidate.Ndis[2],
                        X[:,candidate.nodes], x1[:,g]
                    )
                    if np.linalg.norm(projection[:3]) < 0.27 and -1 <= projection[-1] <= 1:
                        projection = proj.nearest_point_projection(
                            candidate.Ndis[0],
                            candidate.Ndis[1],
                            candidate.Ndis[2],
                            X[:,candidate.nodes], x1[:,g]
                        )
                    projections_all.append(projection)
                except ConvergenceError:
                    projections_all.append(np.ones(4)*np.inf)

            projections_within_domain = [
                i for i in range(len(projections_all)) if -1 <= projections_all[i][-1] <= 1
            ]

            if len(projections_within_domain) == 0:
                self.int_pts[g].activated = False
                return
            
            filtered_projections = [projections_all[i] for i in projections_within_domain]
            filtered_candidates = [candidate_elements[i] for i in projections_within_domain]
            
            distances = np.array([np.linalg.norm(pr[:3]) for pr in filtered_projections])
            selected = np.argmin(distances)
            self.int_pts[g].activated = True
            self.int_pts[g].partner = filtered_candidates[selected]

    def perform_nearest_point_projection(self, X):
        for g in range(len(self.int_pts)):
            partner = self.int_pts[g].partner
            N1 = self.N_displacement[:,g]
            x1 = X[:,self.parent.nodes] @ N1
            try:
                v = proj.nearest_point_projection(
                    partner.Ndis[0],
                    partner.Ndis[1],
                    partner.Ndis[2],
                    X[:,partner.nodes], x1
                )
            except ConvergenceError:
                continue
            self.int_pts[g].s2 = v[-1]
            self.int_pts[g].v = v[:3]
    
    def compute_normal(self):
        for g in range(len(self.int_pts)):
            self.int_pts[g].sigma = np.sign(self.int_pts[g].converged_n2 @ math.normalized(self.int_pts[g].v))
            self.int_pts[g].n2 = self.int_pts[g].sigma * math.normalized(self.int_pts[g].v)
    
    def compute_gap(self):
        try:
            r1 = self.parent.prop.cr
        except AttributeError:
            raise MaterialError("The following data is missing in material specification: 'Contact radius'.")
        for g in range(len(self.int_pts)):
            r2 = self.int_pts[g].partner.prop.cr
            self.int_pts[g].gap = self.int_pts[g].v @ self.int_pts[g].n2 - r1 - r2
        
    def gap_condition_contribution(self, p, X):
        # This function computes elements contribution to node p weak
        #  gap condition value. To determine if node p is active,
        #  this is not enough - contributions from all elements
        #  with p need to be summed.

        # p is global node number - find corresponding local
        find = np.where(self.parent.nodes == p)[0]
        if len(find) == 0: return 0.0
        l = find[0]
        val = 0
        for g in range(len(self.int_pts)):
            if self.int_pts[g].activated:
                gN = self.int_pts[g].gap
                Phi1 = self.N_lagrange[:,g]
                jac = self.parent.jacobian
                val += Phi1[l] * gN * jac * self.int_pts[g].wgt
        return val
        
    def contact_tangent(self, X, Lam, n_nodes_in_mesh):
        n_dof = len(self.dof)
        Kg = np.zeros(shape=(n_dof*n_nodes_in_mesh, n_dof*n_nodes_in_mesh))
        
        for g in range(len(self.int_pts)):
            if self.int_pts[g].activated:
                partner = self.int_pts[g].partner
                n2 = self.int_pts[g].n2
                s2 = self.int_pts[g].s2
                sigma = self.int_pts[g].sigma
                v = self.int_pts[g].v
                v_abs = np.linalg.norm(v)
                if v_abs == 0:
                    print("Error in algorithm - can't process ||v|| == 0. Check equations what should be correct response.")
                Phi1 = self.N_lagrange[:,g]
                N1 = self.N_displacement[:,g]
                lam = Lam[self.parent.nodes] @ Phi1
                N2 = partner.Ndis[0](s2)
                dN2 = partner.Ndis[1](s2)
                ddN2 = partner.Ndis[2](s2)
                dx2 = X[:,partner.nodes] @ dN2
                ddx2 = X[:,partner.nodes] @ ddN2
                S2c = dx2 @ dx2 - v @ ddx2
                jac = self.parent.jacobian
                
                G1 = jac * lam * sigma / v_abs * ((math.skew(n2) @ math.skew(n2)) + np.outer(dx2, dx2) / S2c)
                G2 = jac * lam / S2c * np.outer(dx2, n2)
                G3 = jac * lam / S2c * np.outer(n2, v)

                nodes = (self.parent.nodes, partner.nodes)
                for b1 in range(2):
                    for (i, I) in enumerate(nodes[b1]):
                        for b2 in range(2):
                            for (j, J) in enumerate(nodes[b2]):
                                row_dof = list(range(n_dof*I,n_dof*(I+1)))
                                col_dof = list(range(n_dof*J,n_dof*(J+1)))
                                Kl = np.zeros((n_dof, n_dof))
                                
                                if b1 == 0 and b2 == 0:
                                    Kl[:3,:3] = -N1[i] * G1 * N1[j]
                                    Kl[:3,6] = N1[i] * jac * n2 * Phi1[j]
                                    Kl[6,:3] = Phi1[i] * jac * n2 * N1[j]
                                elif b1 == 0 and b2 == 1:
                                    Kl[:3,:3] = N1[i] * G1 * N2[j] - N1[i] * G2 * dN2[j]
                                    Kl[6,:3] = -Phi1[i] * jac * n2 * N2[j]
                                elif b1 == 1 and b2 == 0:
                                    Kl[:3,:3] = N2[i] * G1 * N1[j] - dN2[i] * G2.T * N1[j]
                                    Kl[:3,6] = -N2[i] * jac * n2 * Phi1[j]
                                elif b1 == 1 and b2 == 1:
                                    Kl[:3,:3] = -N2[i] * G1 * N2[j] + N2[i] * G2 * dN2[j] + dN2[i] * G2.T * N2[j] - dN2[i] * G3 * dN2[j]
                                Kg[np.ix_(row_dof, col_dof)] += self.int_pts[g].wgt * Kl
        return Kg

    def contact_residual(self, X, Lam, n_nodes_in_mesh):
        n_dof = len(self.dof)
        Rg = np.zeros(shape=(n_dof*n_nodes_in_mesh))
        
        for g in range(len(self.int_pts)):
            if self.int_pts[g].activated:
                partner = self.int_pts[g].partner
                n2 = self.int_pts[g].n2
                s2 = self.int_pts[g].s2
                gN = self.int_pts[g].gap
                N1 = self.N_displacement[:,g]
                
                jac = self.parent.jacobian
                Phi1 = self.N_lagrange[:,g]
                lam = Lam[self.parent.nodes] @ Phi1
                N2 = partner.Ndis[0](s2)
                nodes = (self.parent.nodes, partner.nodes)
                for b1 in range(2):
                    for (i, I) in enumerate(nodes[b1]):
                        Rl = np.zeros(n_dof)
                        if b1 == 0:
                            Rl[:3] = N1[i] * jac * lam * n2
                            Rl[6] = Phi1[i] * jac * gN
                        elif b1 == 1:
                            Rl[:3] = -N2[i] * jac * lam * n2
                        row_dof = list(range(n_dof*I,n_dof*(I+1)))
                        Rg[row_dof] += self.int_pts[g].wgt * Rl
        return Rg
    
def Xi_mat(
    dx: np.ndarray,
    dN_displacement: float,
    N_rotation: float,
    dN_rotation: float
) -> np.ndarray:
    Xi = np.identity(6)
    Xi[:3] *= dN_displacement
    Xi[3:] *= dN_rotation
    Xi[3:,:3] = - N_rotation * math.skew(dx)
    return Xi

def assembly_matrix(
    node: int,
    local_dof: np.ndarray,
    n_nodes_in_mesh: int
) -> np.ndarray:
    
    n_loc = np.sum(local_dof)
    n_all = len(local_dof)
    N = n_all * n_nodes_in_mesh
    A = np.zeros(shape=(N,n_loc))
    for j in range(n_loc):
        A[node*n_all+j,j] = 1

    return A
