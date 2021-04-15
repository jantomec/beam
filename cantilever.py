import numpy as np
import elements as el
import matplotlib.pyplot as plt


def cantilever(printing=True):
    if printing: print("Hello, world!\nThis is a FEM program for beam analysis.")

    # INITIALIZATION
    # Let's first load the initial state of the system."
    #print("Looking for a file named init.dat...")
    #print("Found it! Parsing data...")

    # CURRENT STATE
    # All data is written in the form:
    #   variable[0] = last convereged (known) state,
    #   variable[1] = last iteration state
    #   variable[2] = new (converged or not) state
    # Except for time and time_step (which would always just repeat)

    # Constructing a matrix of coordinates    
    # The number of nodes is not allowed to change during the simulation
    global_coordinates = np.zeros(shape=(3,6))
    global_coordinates[0] = np.linspace(0,1,num=6)
    n_dim = global_coordinates.shape[0]
    n_nodes = global_coordinates.shape[1]
    
    # The number of dofs per node in mesh is not allowed to change
    #  during the simulation. They can be, however, actived or
    #  deactivated
    n_dof = 7
    
    # Constructing a vector of unknowns
    # vector of unknowns for the solver A.x = b
    x = np.zeros(shape=(n_dof,n_nodes))
    
    # Constructing matrices of displacement, velocity and acceleration
    global_displacements = [None, None, np.zeros((n_dim,n_nodes))]
    global_velocities = [None, None, np.zeros((n_dim,n_nodes))]
    global_accelerations = [None, None, np.zeros((n_dim,n_nodes))]

    history = []

    # Numbering elements  
    def ele_nodes(ele_id, n_nodes_per_ele):
        return np.array([
            n_nodes_per_ele*ele_id+j for j in range(n_nodes_per_ele+1)
        ], dtype=int)
    
    elements = [
        None,
        None, [
            el.SimoBeam(
                index=i,
                nodes=ele_nodes(i, 1),
                n_nodes_in_mesh=n_nodes,
                mesh_dof_per_node=n_dof,
                ref_vec=np.array([0,0,1]),
                coordinates=global_coordinates[:,ele_nodes(i, 1)],
                area=1.0,
                elastic_modulus=1.0,
                shear_modulus=1.0,
                inertia_primary=2.0,
                inertia_secondary=1.0,
                inertia_torsion=1.0
            ) for i in range(5)
        ]
    ]

    # Determining the active degrees of freedom  
    # Variable doesn't change with Newton iterations, only with time 
    #  step.
    active_dof = [
        None,
        np.ones((n_dof,n_nodes), dtype=np.bool)
    ]
    active_dof[1][6] = False
    active_dof[1][:,0] = False
    
    # Add force and/or displacements loads    
    def Qload(t):
        Q = np.zeros(shape=(6, n_nodes))
        Q[4,-1] = 8*np.pi
        return Q
    Qfollow = lambda t : np.zeros_like(Qload)
    Uload = lambda t : np.zeros(shape=(6, n_nodes))
    
    # Note that additional nodal values can be added if
    #  necessary simply by creating new matrices.

    if printing: print("Initiating time...")

    # Variable doesn't change with Newton iterations, only with time 
    #  step.
    time_step = [None, 1.]
    time = [None, 0.]
    final_time = 1.
    
    # Selecting solver parameters    
    max_number_of_time_steps = 1 #  100
    max_number_of_newton_iterations = 5
    tolerance = 1e-8
    conv_test = "RES"
    # conv_test = "DSP"
    # conv_test = "ENE"
    
    if printing: print("We use Newmark-beta method.")
    beta = 0.25
    gamma = 0.5
    def matrix_multipliers():

        # if static
        c = (1.0, 0.0, 0.0)

        # if dynamic
        # c = (1.0, gamma/(time_step[1]*beta), 1/(time_step[1]**2*beta))
        
        return c

    if printing: print("Start of time loop...")

    for n in range(max_number_of_time_steps):
        if printing: print("Step", n, "\n\tTime", time[1])

        time_step[0] = time_step[1]
        time[0] = time[1]

        # New time step and new time.
        c = matrix_multipliers()
        time[1] = time[0] + time_step[1]

        elements[0] = elements[2]
        global_displacements[0] = global_displacements[2]
        history.append(
            global_coordinates +
            global_displacements[0]
        )
        global_velocities[0] = global_velocities[2]
        global_accelerations[0] = global_accelerations[2]
        active_dof[0] = active_dof[1]
        
        # Apply displacement load
        x[:6] = Uload(time[1])

        for i in range(max_number_of_newton_iterations):
            tangent = np.zeros((n_dof*n_nodes, n_dof*n_nodes))

            elements[1] = elements[2]
            global_displacements[1] = global_displacements[2]
            global_velocities[1] = global_velocities[2]
            global_accelerations[1] = global_accelerations[2]

            if i > 0:
                # Assembly of tangent matrix from all elements.
                for e in range(len(elements[2])):
                    if type(elements[2][e]) == el.SimoBeam:
                        S = c[0] * elements[2][e].stiffness_matrix(
                            global_coordinates[:,elements[2][e].nodes] +
                            global_displacements[2][
                                :,
                                elements[2][e].nodes
                            ]
                        )
                        if c[2] != 0:
                            S += c[2] * elements[2][e].mass_matrix(
                                time_step[1], beta, gamma
                            )
                        A = elements[2][e].assemb
                        tangent += A @ S @ A.T
                        
                # Solve system of equations.
                mask = ~active_dof[1].flatten('F')
                tangent[mask] = np.identity(n_dof*n_nodes)[mask]
                tangent[:,mask] = np.identity(n_dof*n_nodes)[:,mask]
                x = x.flatten('F')
                x[mask] = np.zeros(shape=(n_dof*n_nodes))[mask]
                x = np.linalg.solve(tangent, x)
                x = np.reshape(x, newshape=(n_dof,n_nodes), order='F')

            # Update values.
            global_displacements[2] += x[:3]
            if i == 0:
                a_new = (
                    (1 - 0.5/beta) * global_accelerations[2] -
                    1/(time_step[1]*beta) * global_velocities[2]
                )
                global_velocities[2] += time_step[1] * (
                    (1 - gamma) * global_accelerations[2] +
                    gamma * a_new
                )
                global_accelerations[2] = a_new
            else:
                global_velocities[2] += (
                    gamma / 
                    (time_step[1] * beta) * x[:3]
                )
                global_accelerations[2] += (
                    1 / 
                    (time_step[1]**2 * beta) * x[:3]
                )
            for e in range(len(elements[2])):
                if type(elements[2][e]) == el.SimoBeam:
                    elements[2][e].update(
                        global_coordinates[:,elements[2][e].nodes] +
                        global_displacements[2][:,elements[2][e].nodes],
                        x[3:6,elements[2][e].nodes],
                        time_step[1],
                        beta,
                        gamma,
                        iter0=(i == 0)
                    )
            
            # Displacement convergence
            if conv_test == "DSP" and np.linalg.norm(x) <= tolerance:
                if printing: print("Time step converged within", i+1, "iterations.")
                break
            
            # External forces
            x[:6] = Qload(time[1])

            # Internal forces
            for e in range(len(elements[2])):
                if type(elements[2][e]) == el.SimoBeam:
                    R = c[0] * elements[2][e].stiffness_residual(
                        global_coordinates[:,elements[2][e].nodes] +
                        global_displacements[2][:,elements[2][e].nodes]
                    )
                    if c[2] != 0:
                        R += elements[2][e].mass_residual(
                            global_accelerations[2][
                                :,
                                elements[2][e].nodes
                            ]
                        )
                    A = elements[2][e].assemb
                    x -= np.reshape(
                        A @ R,
                        newshape=(n_dof, n_nodes),
                        order='F'
                    )
            
            # Residual convergence
            res_norm = np.linalg.norm(x[active_dof[1]])
            if conv_test == "RES" and res_norm <= tolerance:
                if printing: print("\tTime step converged within", i+1, "iterations.\n")
                break

        else:
            if printing: print("\nMaximum number of iterations reached without convergence!")
            return
        
        if time[1] >= final_time:
            if printing: print("Computation is finished, reached the end of time.")
            history.append(
                global_coordinates + global_displacements[2]
            )
            h = np.array(history)
            return h

    if printing: print("Final time was never reached.")
    return

def main():
    h = cantilever()
    plt.plot(h[-1,0], h[-1,2])
    plt.show()

if __name__ == "__main__":
    main()
