import numpy as np
import elements as el
import matplotlib.pyplot as plt
import contact


def cantilever_contact(printing=True):
    if printing: print("Hello, world!\nThis is a FEM program for beam analysis.")

    # INITIALIZATION
    # Let's first load the initial state of the system
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
    
    n_ele_1 = 10
    ord_1 = 1
    n_nod_1 = ord_1*n_ele_1+1
    n_ele_2 = 1
    ord_2 = 1
    n_nod_2 = ord_2*n_ele_2+1
    coordinates = np.zeros((3,n_nod_1+n_nod_2))
    coordinates[0,:n_nod_1] = np.linspace(0,100,n_nod_1)
    coordinates[2,:n_nod_1] = 10
    coordinates[0,n_nod_1:n_nod_1+n_nod_2] = np.linspace(0,100,n_nod_2)
    n_dim = coordinates.shape[0]
    n_nodes = coordinates.shape[1]
    n_dof = 7
    
    # Constructing a vector of unknowns
    # vector of unknowns for the solver A.x = b
    x = np.zeros(shape=(n_dof,n_nodes))
    
    # Constructing matrices of displacements, velocities and accelerations
    displacement = np.zeros((3,n_dim,n_nodes))
    velocity = np.zeros((3,n_dim,n_nodes))
    acceleration = np.zeros((3,n_dim,n_nodes))
    lagrange = np.zeros((3,n_nodes))

    history = []

    # Numbering elements
    def ele_nodes(ele_id, n_nodes_per_ele):
        return np.array([
            n_nodes_per_ele*ele_id+j for j in range(n_nodes_per_ele+1)
        ], dtype=int)
    
    beam_1 = []
    for i in range(n_ele_1):
        element_on_beam_1 = el.SimoBeam(
            nodes=ele_nodes(i, ord_1),
            n_nodes_in_mesh=n_nodes,
            mesh_dof_per_node=n_dof,
            ref_vec=np.array([0,0,1]),
            coordinates=coordinates[:,ele_nodes(i, ord_1)],
            beam=0,
            area=10.0,
            elastic_modulus=10.0,
            shear_modulus=10.0,
            inertia_primary=20.0,
            inertia_secondary=10.0,
            inertia_torsion=10.0,
            density=10.0,
            contact_radius=4
        )
        element_on_beam_1.child = el.MortarContact(
            parent_element=element_on_beam_1,
            n_integration_points=2
        )
        beam_1.append(element_on_beam_1)

    beam_2 = []
    for i in range(n_ele_2):
        element_on_beam_2 = el.SimoBeam(
            nodes=ele_nodes(i, ord_2)+n_nod_1,
            n_nodes_in_mesh=n_nodes,
            mesh_dof_per_node=n_dof,
            ref_vec=np.array([0,0,1]),
            coordinates=coordinates[:,ele_nodes(i, ord_2)+n_nod_1],
            beam=1,
            area=10.0,
            elastic_modulus=10.0,
            shear_modulus=10.0,
            inertia_primary=20.0,
            inertia_secondary=10.0,
            inertia_torsion=10.0,
            density=10.0,
            contact_radius=4
        )
        beam_2.append(element_on_beam_2)

    elements = beam_1 + beam_2
    mortar_elements = beam_2
    mortar_nodes = contact.collect_nodes(mortar_elements)

    # Determine the active degrees of freedom
    # Variable doesn't change with Newton iterations, only with time 
    #  step.
    active_dof = np.ones((2,n_dof,n_nodes), dtype=np.bool)
    active_dof[1][6] = False
    active_dof[1][:,0] = False
    active_dof[1][:,n_nod_1] = False
    # Add force and/or displacements loads
    def Qload(t):
        Q = np.zeros(shape=(6, n_nodes))
        Q0 = -np.pi/20 / 150
        T = 100
        if t <= T:
            Q[2,n_nod_1-1] = Q0 * t
            Q[2,-1] = -Q0 * t
        else:
            Q[2,n_nod_1-1] = T*Q0
            Q[2,-1] = -T*Q0
        return Q
    Qfollow = lambda t : np.zeros_like(Qload)
    Uload = lambda t : np.zeros(shape=(6, n_nodes))
    
    # Note that additional nodal values can be added if necessary simply by creating new matrices

    # Initiating time
    # Variable doesn't change with Newton iterations, only with time 
    #  step.
    time_step = [None, 2.0]
    time = [None, 0.]
    final_time = 110.
    
    # Select solver parameters
    max_number_of_time_steps = 300 #  100
    max_number_of_newton_iterations = 60
    max_number_of_contact_iterations = 10
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
    
    # Start of time loop
    for n in range(max_number_of_time_steps):
        if printing: print("Step", n, "\n\tTime", time[1])

        ################################################################
        # Initiate time step
        time_step[0] = time_step[1]
        time[0] = time[1]

        c = matrix_multipliers()
        time[1] = time[0] + time_step[1]

        displacement[0] = displacement[2]
        history.append(coordinates+displacement[0])
        velocity[0] = velocity[2]
        acceleration[0] = acceleration[2]
        lagrange[0] = lagrange[2]
        
        ################################################################
        # Contact search
        active_nodes_changed = True
        for b1 in beam_1:
            try:
                b1.child.find_partner(coordinates+displacement[2], mortar_nodes, mortar_elements)
            except:
                return finish_computation(beam_1, history, coordinates, displacement[2])

        contact_loop_counter = 0
        while active_nodes_changed:
            active_nodes_changed = False
            # Apply displacement load
            x[:] = 0.0
            x[:6] = Uload(time[1])
            ############################################################
            # Newton-Raphson iteration
            for i in range(max_number_of_newton_iterations):
                # ------------------------------------------------------
                # Initiate new iteration
                tangent = np.zeros((n_dof*n_nodes, n_dof*n_nodes))

                displacement[1] = displacement[2]
                velocity[1] = velocity[2]
                acceleration[1] = acceleration[2]
                lagrange[1] = lagrange[2]

                if i > 0:
                    # --------------------------------------------------
                    # Assembly of tangent matrix from all elements.
                    mask = ~active_dof[1].flatten('F')

                    for e in range(len(elements)):
                        if type(elements[e]) == el.SimoBeam:
                            S = c[0] * elements[e].stiffness_matrix(
                                coordinates[:,elements[e].nodes] +
                                displacement[2][:,elements[e].nodes]
                            )
                            if c[2] != 0:
                                S += c[2] * elements[e].mass_matrix(
                                    time_step[1], beta, gamma
                                )
                            A = elements[e].assemb
                            tangent += A @ S @ A.T
                    
                    # Contact contributions
                    for b1 in beam_1:
                        tangent += c[0] * b1.child.contact_tangent(
                            coordinates+displacement[2], lagrange[2], n_nodes
                        )
                        
                    # Follower load contributions
                    # ---

                    # --------------------------------------------------        
                    # Solve system of equations.
                    tangent[mask] = np.identity(n_dof*n_nodes)[mask]
                    tangent[:,mask] = np.identity(n_dof*n_nodes)[:,mask]
                    x = x.flatten('F')
                    x[mask] = np.zeros(shape=(n_dof*n_nodes))[mask]
                    x = np.linalg.solve(tangent, x)
                    x = np.reshape(x, newshape=(n_dof,n_nodes), order='F')

                # ------------------------------------------------------
                # Update beam values
                displacement[2] += x[:3]
                if i == 0:
                    a_new = (
                        (1 - 0.5/beta) * acceleration[2] -
                        1/(time_step[1]*beta) * velocity[2]
                    )
                    velocity[2] += time_step[1] * (
                        (1 - gamma) * acceleration[2] +
                        gamma * a_new
                    )
                    acceleration[2] = a_new
                else:
                    velocity[2] += (
                        gamma / 
                        (time_step[1] * beta) * x[:3]
                    )
                    acceleration[2] += (
                        1 / 
                        (time_step[1]**2 * beta) * x[:3]
                    )
                for e in range(len(elements)):
                    if type(elements[e]) == el.SimoBeam:
                        elements[e].update(
                            coordinates[:,elements[e].nodes] +
                            displacement[2][:,elements[e].nodes],
                            x[3:6,elements[e].nodes],
                            time_step[1],
                            beta,
                            gamma,
                            iter0=(i == 0)
                        )
                
                # Update contact values
                lagrange[2] += x[6]
                
                # ------------------------------------------------------
                # Displacement convergence
                if conv_test == "DSP" and np.linalg.norm(x) <= tolerance:
                    if printing: print("Time step converged within", i+1, "iterations.")
                    break
                
                # ------------------------------------------------------
                # External forces
                x[:6] = Qload(time[1])

                # Internal forces
                for e in range(len(elements)):
                    if type(elements[e]) == el.SimoBeam:
                        R = c[0] * elements[e].stiffness_residual(
                            coordinates[:,elements[e].nodes] +
                            displacement[2][:,elements[e].nodes]
                        )
                        if c[2] != 0:
                            R += elements[e].mass_residual(
                                acceleration[2][:,elements[e].nodes]
                            )
                        A = elements[e].assemb
                        x -= np.reshape(
                            A @ R,
                            newshape=(n_dof, n_nodes),
                            order='F'
                        )
                
                # Contact forces
                for b1 in beam_1:
                    try:
                        b1.child.find_gap(coordinates+displacement[2])
                    except:
                        print("Algorithm was distrupted by error in gap computation.")
                        return finish_computation(beam_1, history, coordinates, displacement[2])
                    contact_forces = np.reshape(
                        b1.child.contact_residual(coordinates+displacement[2], lagrange[2], n_nodes),
                        newshape=(n_dof, n_nodes),
                        order='F'
                    )
                    x -= contact_forces
                    
                # ------------------------------------------------------
                # Residual convergence
                res_norm = np.linalg.norm(x[active_dof[1]])
                # print("Residual", res_norm)
                if conv_test == "RES" and res_norm <= tolerance:
                    if printing: print("\tTime step converged within", i+1, "iterations.\n")
                    break

            else:
                print("\nMaximum number of iterations reached without "
                    + "convergence!")
                return finish_computation(beam_1, history, coordinates, displacement[2])
        
            ############################################################
            # Continue contact
            # Newton-Raphson method converged to a new solution
            
            active_dof[0] = active_dof[1]

            # Check contact conditions
            # Inactive nodes
            for p in range(n_nodes):
                if active_dof[1][6,p] == False:
                    gap_condition_for_node_p = 0.0
                    for element in beam_1:
                        X = coordinates + displacement[2]
                        gap_condition_for_node_p += element.child.gap_condition_contribution(p, X)
                    if gap_condition_for_node_p < 0:
                        active_dof[1][6,p] = True
                else:
                    pressure_condition_for_node_p = 0.0
                    for element in beam_1:
                        X = coordinates + displacement[2]
                        pressure_condition_for_node_p += element.child.pressure_condition_contribution(p, X, lagrange[2])
                    
                    if pressure_condition_for_node_p > 0:
                        active_dof[1][6,p] = False
                        lagrange[2][p] = 0.0
            active_nodes_changed = not np.all(active_dof[0][6] == active_dof[1][6])
            if printing and active_nodes_changed: print("\tActive nodes have changed: repeat time step.\n")
            contact_loop_counter += 1
            if contact_loop_counter > max_number_of_contact_iterations:
                print("\nContact: Maximum number of contact iterations",
                      "reached without convergence!")
                return finish_computation(beam_1, history, coordinates, displacement[2])

        ################################################################
        # Finish time step
        if time[1] >= final_time:
            if printing: print("Computation is finished, reached the end of time.")
            history.append(coordinates+displacement[0].copy())
            return finish_computation(beam_1, history, coordinates, displacement[2])

    if printing: print("Final time was never reached.")
    return finish_computation(beam_1, history, coordinates, displacement[2])

def finish_computation(beam_1, history, coordinates, displacement):
    X = coordinates + displacement
    gap_function = []
    for b1 in beam_1:
        for g in range(len(b1.child.int_pts)):
            x = X[0,b1.nodes] @ b1.child.N_displacement[:,g]
            y = b1.child.int_pts[g].gap
            gap_function.append([x,y])
    gap_function = np.array(gap_function)
    h = np.array(history)
    return (h, gap_function)

def main():
    np.set_printoptions(linewidth=300, floatmode='fixed', precision=5)
    (h, gap_function) = cantilever_contact()
    n_nod_1 = np.argmin(h[0,0,1:])+1
    color_map = plt.get_cmap("tab10")
    c0 = color_map(0)
    c1 = color_map(1)
    if type(h) == np.ndarray:
        for i in range(1,len(h),1):
            plt.plot(h[0,0,:n_nod_1],h[0,2,:n_nod_1], '-', linewidth=6.0, color=c0, alpha=0.5)
            plt.plot(h[0,0,n_nod_1:],h[0,2,n_nod_1:], '-', linewidth=6.0, color=c0, alpha=0.5)
            plt.plot(h[0,0,:n_nod_1],h[0,2,:n_nod_1], '.-', color=c0)
            plt.plot(h[0,0,n_nod_1:],h[0,2,n_nod_1:], '.-', color=c0)
            plt.plot(h[i,0,:n_nod_1],h[i,2,:n_nod_1], '-', linewidth=6.0, color=c1, alpha=0.5)
            plt.plot(h[i,0,n_nod_1:],h[i,2,n_nod_1:], '-', linewidth=6.0, color=c1, alpha=0.5)
            plt.plot(h[i,0,:n_nod_1],h[i,2,:n_nod_1], '.-', label=i, color=c1)
            plt.plot(h[i,0,n_nod_1:],h[i,2,n_nod_1:], '.-', color=c1)
            plt.xlim((-20,130))
            plt.ylim((-30,70))
            plt.legend()
            plt.show()
    plt.plot(gap_function[:,0], gap_function[:,1], 'o')
    plt.show()
if __name__ == "__main__":
    main()
