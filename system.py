import numpy as np
from typing import Callable
import interpolation as intp
from errors import ConvergenceError


class System:
    def __init__(self, coordinates, elements):
        # Mesh
        assert coordinates.shape[0] == 3, 'Only three-dimensional systems are currently supported.'
        self.coordinates = coordinates
        self.elements = elements
        self.degrees_of_freedom = [np.ones((7,coordinates.shape[1]), dtype=np.bool)]
        self.degrees_of_freedom[0][6,:] = False  # defualt - no contact at the beginning
        
        # Constructing a vector of unknowns for the solver A.x = b
        self.unknowns = np.zeros_like(self.degrees_of_freedom[0], dtype=np.float)

        # Constructing matrices of physical fields
        n_nodes = self.coordinates.shape[1]
        self.displacement = [np.zeros((3,n_nodes))]
        self.velocity = [np.zeros((3,n_nodes))]
        self.acceleration = [np.zeros((3,n_nodes))]
        self.lagrange = [np.zeros(n_nodes)]
        
        # Initiating time (variable doesn't change with Newton iterations, only with time step)
        self.time_step = 1.0
        self.time = [0.0]
        self.__time = self.time[0]
        self.final_time = 1.0

        # Select solver parameters
        self.max_number_of_time_steps = 100
        self.max_number_of_newton_iterations = 60
        self.max_number_of_contact_iterations = 10
        self.tolerance = 1e-8
        self.convergence_test_type = "RES"  # options: "RES" - force residual, "DSP" - displacement, "ENE" - energy
        self.solver_type = "dynamic"  # options: "dynamic", "static"
        self.dynamic_solver_type = "Newmark-beta method"  # options: "Newmark-beta method", "Generalized-alpha method"
        self.dynamic_update_variant = "True velocity"  # options: "True velocity", "Iterative velocity"
        self.beta = 0.25
        self.gamma = 0.5
        self.contact_detection = True
        
        # Other parameters
        self.printing = True

    def get_number_of_nodes(self):
        return self.coordinates.shape[1]
    def get_number_of_elements(self):
        return len(self.elements)
    def get_number_of_degrees_of_freedom_per_node(self):
        return self.degrees_of_freedom[-1].shape[0]
    def get_number_of_all_degrees_of_freedom(self):
        return len(self.degrees_of_freedom[-1].flatten(order='F'))
    def get_matrix_multipliers(self):
        if self.solver_type == 'static':
            return np.array([1.0, 0.0, 0.0])
        elif self.solver_type == 'dynamic':
            return np.array([1.0, self.gamma/(self.time_step*self.beta), 1/(self.time_step**2*self.beta)])
        else:
            raise Exception('Solver parameter error - only "static" or "dynamic" types are supported for solver_type.')
    
    def force_load(self):
        n_nodes = self.get_number_of_nodes()
        return np.zeros((6, n_nodes))
    
    def follower_force_load(self):
        n_nodes = self.get_number_of_nodes()
        return np.zeros((6, n_nodes))
    
    def displacement_load(self):
        n_nodes = self.get_number_of_nodes()
        return np.zeros((6, n_nodes))

    def __newton_loop(self):
        c = self.get_matrix_multipliers()
        n_dof = self.get_number_of_all_degrees_of_freedom()
        n_ndof = self.get_number_of_degrees_of_freedom_per_node()
        n_nodes = self.get_number_of_nodes()
        n_ele = self.get_number_of_elements()
        
        x = self.unknowns

        # Apply displacement load
        x[:] = 0.0
        x[:6] = self.displacement_load()

        # Perform Newton-Raphson iteration method to find new balance
        for i in range(self.max_number_of_newton_iterations):
            # Initiate new iteration
            self.__displacement[1] = self.__displacement[2]
            self.__velocity[1] = self.__velocity[2]
            self.__acceleration[1] = self.__acceleration[2]
            self.__lagrange[1] = self.__lagrange[2]

            if i > 0:
                # Assembly of tangent matrix from all elements.
                tangent = np.zeros((n_dof, n_dof))
                mask = ~self.__degrees_of_freedom[1]
                for ele in self.elements:
                    A = ele.assemb

                    # Static contribution
                    X = self.coordinates[:,ele.nodes] + self.__displacement[2][:,ele.nodes]
                    S = c[0] * ele.stiffness_matrix(X)
                    tangent += A @ S @ A.T

                    # Dynamic contribution
                    if c[2] != 0:
                        S = c[2] * ele.mass_matrix(self.time_step, self.beta, self.gamma)
                        tangent += A @ S @ A.T
                    
                    # Follower load contributions
                    # ---

                    # Contact contribution
                    try:
                        contact_element = ele.child
                        tangent += c[0] * contact_element.contact_tangent(self.coordinates+self.__displacement[2], self.__lagrange[2], n_nodes)
                    except AttributeError:
                        pass
                    
                # Solve system of equations.
                tangent[mask.flatten(order='F')] = np.identity(n_dof)[mask.flatten(order='F')]
                tangent[:,mask.flatten(order='F')] = np.identity(n_dof)[:,mask.flatten(order='F')]
                x[mask] = np.zeros(shape=(n_ndof,n_nodes))[mask]
                x = np.linalg.solve(tangent, x.flatten(order='F')).reshape((n_ndof,n_nodes), order='F')
            
            # Update nodal beam values
            self.__displacement[2] += x[:3]
            if i == 0:
                a_new = (
                    (1 - 0.5/self.beta) * self.__acceleration[2] -
                    1/(self.time_step*self.beta) * self.__velocity[2]
                )
                self.__velocity[2] += self.time_step * (
                    (1 - self.gamma) * self.__acceleration[2] +
                    self.gamma * a_new
                )
                self.__acceleration[2] = a_new
            else:
                if self.dynamic_update_variant == "True velocity":
                    displacement_change = self.__displacement[2] - self.__displacement[0]
                elif self.dynamic_update_variant == "Iterative velocity":
                    displacement_change = x[:3]
                else:
                    raise Exception('Solver parameter error - only "True velocity" or "Iterative velocity" types are supported for dynamic_update_variant.')
                
                self.__velocity[2] += (
                    self.gamma / 
                    (self.time_step * self.beta) * displacement_change
                )
                self.__acceleration[2] += (
                    1 / 
                    (self.time_step**2 * self.beta) * displacement_change
                )
            # Update integration point beam values
            for ele in self.elements:
                X = self.coordinates[:,ele.nodes] + self.__displacement[2][:,ele.nodes]
                ele.update(X, x[3:6,ele.nodes], self.time_step, self.beta, self.gamma, iter0=(i == 0))
            
            # Update nodal contact values
            self.__lagrange[2] += x[6]

            # Update integration point contact values
            for ele in self.elements:
                try:
                    contact_element = ele.child
                    contact_element.find_gap(self.coordinates+self.__displacement[2], self.__velocity[2])
                except AttributeError:
                    pass
            
            # Displacement convergence
            if self.convergence_test_type == "DSP" and np.linalg.norm(x) <= self.tolerance:
                if self.printing: print("Time step converged within", i+1, "iterations.")
                break
            
            # External forces
            x[:6] = self.force_load()

            for ele in self.elements:
                # Internal forces
                X = self.coordinates[:,ele.nodes] + self.__displacement[2][:,ele.nodes]
                R = c[0] * ele.stiffness_residual(X)
                if c[2] != 0:
                    R += ele.mass_residual(self.__acceleration[2][:,ele.nodes])
                A = ele.assemb
                x -= (A @ R).reshape((n_ndof, n_nodes), order='F')
            
                # Contact forces
                try:
                    contact_element = ele.child
                    contact_forces = contact_element.contact_residual(self.coordinates+self.__displacement[2], self.__lagrange[2], n_nodes).reshape((n_dof, n_nodes), order='F')
                    x -= contact_forces
                except AttributeError:
                    pass
            
            # Residual convergence
            res_norm = np.linalg.norm(x[self.__degrees_of_freedom[1]])
            # if printing: print("Residual", res_norm)
            if self.convergence_test_type == "RES" and res_norm <= self.tolerance:
                # Newton-Raphson algorithm converged to a new solution
                self.__degrees_of_freedom[0] = self.__degrees_of_freedom[1]
                if self.printing: print("\tTime step converged within", i+1, "iterations.\n")
                break

        else:
            raise ConvergenceError('Newton-Raphson: Maximum number of iterations reached without convergence.')

    def __contact_loop(self):
        n_nodes = self.coordinates.shape[1]
        x = self.unknowns
        self.__degrees_of_freedom = self.__degrees_of_freedom

        active_nodes_changed = True
        for ele in self.elements:
            try:
                contact_element = ele.child
                contact_element.find_partner(self.coordinates+self.__displacement[2])
            except AttributeError:
                pass

        contact_loop_counter = 0
        while active_nodes_changed:
            active_nodes_changed = False
            
            # Newton-Raphson method converged to a new solution
            self.__newton_loop()
            
            # Check contact conditions
            # Inactive nodes
            for p in range(n_nodes):
                if self.__degrees_of_freedom[1][6,p] == False:
                    gap_condition_for_node_p = 0.0
                    for ele in self.elements:
                        try:
                            contact_element = ele.child
                            X = self.coordinates + self.__displacement[2]
                            gap_condition_for_node_p += contact_element.gap_condition_contribution(p, X)
                        except AttributeError:
                            continue
                    if gap_condition_for_node_p < 0:
                        self.__degrees_of_freedom[1][6,p] = True
                else:
                    pressure_condition_for_node_p = 0.0
                    for ele in self.elements:
                        try:
                            contact_element = ele.child
                            X = self.coordinates + self.__displacement[2]
                            pressure_condition_for_node_p += contact_element.pressure_condition_contribution(p, X, self.__lagrange[2])
                        except AttributeError:
                            continue
                    
                    if pressure_condition_for_node_p > 0:
                        self.__degrees_of_freedom[1][6,p] = False
                        self.__lagrange[2][p] = 0.0
            active_nodes_changed = np.any(self.__degrees_of_freedom[0][6] != self.__degrees_of_freedom[1][6])
            if self.printing and active_nodes_changed: print("\tActive nodes have changed: repeat time step.\n")
            contact_loop_counter += 1
            if contact_loop_counter > self.max_number_of_contact_iterations:
                raise ConvergenceError("Contact: Maximum number of contact iterations reached without convergence!")

    def __time_loop(self):
        # Start of time loop
        for n in range(self.max_number_of_time_steps):
            if self.printing: print("Time step: ", n+1, " (time ", self.__time, " --> ", self.__time + self.time_step, ")", sep='')

            self.__time += self.time_step
            self.__displacement[0] = self.__displacement[2]
            self.__velocity[0] = self.__velocity[2]
            self.__acceleration[0] = self.__acceleration[2]
            self.__lagrange[0] = self.__lagrange[2]
            
            if self.contact_detection:
                self.__contact_loop()
            else:
                self.__newton_loop()
            
            self.displacement.append(self.__displacement[2].copy())
            self.velocity.append(self.__velocity[2].copy())
            self.acceleration.append(self.__acceleration[2].copy())
            self.lagrange.append(self.__lagrange[2].copy())
            self.time.append(self.__time)

            if self.__time >= self.final_time:
                if self.printing: print("Computation is finished, reached the end of time.")
                break
    
    def postprocessor(self):
        print("Post-proccessing method must be overriden.")
        return

    def gap_function(self, axis=0):
        """
        Return gap function values along one of the main axes.
        """
        gaps = []
        X = self.coordinates + self.displacement[-1]
        for ele in self.elements:
            try:
                contact_element = ele.child
                for g in range(len(contact_element.int_pts)):
                    x = X[axis,ele.nodes] @ contact_element.N_displacement[:,g]
                    y = contact_element.int_pts[g].gap
                    gaps.append([x,y])
            except AttributeError:
                continue
        gaps = np.array(gaps)
        return gaps

    def solve(self):
        if self.printing:
            print("Hello, world!\nThis is a FEM program for beam analysis.")
            print("This will be", self.solver_type, "analysis.")
            print("We use", self.dynamic_solver_type + ".")
        
        self.__degrees_of_freedom = np.array([self.degrees_of_freedom[-1]]*2)
        self.__displacement = np.array([self.displacement[-1]]*3)
        self.__velocity = np.array([self.velocity[-1]]*3)
        self.__acceleration = np.array([self.acceleration[-1]]*3)
        self.__lagrange = np.array([self.lagrange[-1]]*3)
        self.__time_loop()
