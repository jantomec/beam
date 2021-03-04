import numpy as np
import elements as el


def main():
    print("Hello, world!")
    print("This is a FEM program for beam analysis.")

    # INITIALIZATION
    print("Let's first load the initial state of the system.")
    #print("Looking for a file named init.dat...")
    #print("Found it! Parsing data...")

    # CURRENT STATE
    # All data is written in the form:
    #   variable[0] = last convereged (known) state,
    #   variable[1] = last iteration state
    #   variable[2] = new (converged or not) state
    # Except for time and time_step (which would always just repeat)

    print("Constructing a matrix of coordinates...")
    # The number of nodes is not allowed to change during the simulation
    n_nodes = 6
    n_dim = 3
    global_coordinates = np.zeros(shape=(n_dim,n_nodes))
    global_coordinates[0] = np.linspace(0,1,num=n_nodes)
    
    # The number of dofs per node in mesh is not allowed to change
    #  during the simulation. They can be, however, actived or
    #  deactivated
    n_dof = 7
    
    print("Constructing a vector of unknowns...")
    # vector of unknowns for the solver A.x = b
    x = np.zeros(shape=(n_dof,n_nodes))
    
    print("Constructing matrices of displacements, " + 
          "velocities and accelerations...")
    global_displacements = [None, None, np.zeros((n_dim,n_nodes))]
    global_velocities = [None, None, np.zeros((n_dim,n_nodes))]
    global_accelerations = [None, None, np.zeros((n_dim,n_nodes))]

    print("Numbering elements...")
    elements = [
        None,
        None,
        [
            el.SimoBeam(
                nodes=[i,i+1],
                n_nodes_in_mesh=n_nodes,
                mesh_dof_per_node=n_dof,
                ref_vec=np.array([0,0,1]),
                coordinates=global_coordinates[:,[i,i+1]],
                area=1.0,
                elastic_modulus=1.0,
                shear_modulus=1.0,
                inertia_primary=2.0,
                inertia_secondary=1.0,
                inertia_torsion=1.0
            ) for i in range(5)
        ]
    ]

    print("Determining the active degrees of freedom...")
    # Variable doesn't change with Newton iterations, only with time 
    #  step.
    active_dof = [
        None,
        np.ones((n_dof,n_nodes), dtype=np.bool)
    ]
    active_dof[1][6] = False
    active_dof[1][:,0] = False
    
    print("Add force and/or displacements loads...")
    def Qload(t):
        Q = np.zeros(shape=(6, n_nodes))
        Q[4,-1] = 8*np.pi
        return Q
    Qfollow = lambda t : np.zeros_like(Qload)
    Uload = lambda t : np.zeros(shape=(6, n_nodes))
    
    print("Note that additional nodal values can be added if necessary "
          + "simply by creating new matrices.")

    print("Initiating time...")
    # Variable doesn't change with Newton iterations, only with time 
    #  step.
    time_step = [None, 1.]
    time = [None, 0.]
    final_time = 1.
    
    print("Selecting solver parameters...")
    max_number_of_time_steps = 4 #  100
    max_number_of_newton_iterations = 10
    tolerance = 1e-8
    conv_test = "RES"
    # conv_test = "DSP"
    # conv_test = "ENE"
    
    print("We use Newmark-beta method.")
    beta = 0.25
    gamma = 0.5
    c = (1.0, 0.0, 0.0)  # if static
    # c = (1.0, gamma/(dt*beta), 1/(dt**2*beta))  # if dynamic

    print("Start of time loop...")
    for n in range(max_number_of_time_steps):
        time_step[0] = time_step[1]
        time[0] = time[1]

        # New time step and new time.
        time_step[1] = time_step[0]
        time[1] = time[0] + time_step[1]

        elements[0] = elements[2]
        global_displacements[0] = global_displacements[2]
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
                            global_displacements[2][:,elements[2][e].nodes]
                        )
                        if c[1] != 0:
                            S += c[2] * elements[2][e].mass_matrix()
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
            # global_velocities[2]
            # global_accelerations[2]
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
                print("Time step converged within", i+1, "iterations.")
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
                    if c[1] != 0:
                        R += c[2] * elements[2][e].mass_residual()
                    A = elements[2][e].assemb
                    x -= np.reshape(A @ R, newshape=(n_dof, n_nodes), order='F')
            
            # Residual convergence
            res_norm = np.linalg.norm(x[active_dof[1]])
            if conv_test == "RES" and res_norm <= tolerance:
                print("Time step converged within", i+1, "iterations.")
                break

        else:
            print("Maximum number of iterations reached without "
                  + "convergence!")
            return
        
        if time[1] >= final_time:
            print("Computation is finished, reached the end of time.")
            return

    print("Final time was never reached.")
    return

if __name__ == "__main__":
    main()
