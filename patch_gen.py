import numpy as np
import elements as el
import matplotlib.pyplot as plt
import contact as ct


def patch():
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
    
    n_ele_1 = 6
    ord_1 = 1
    n_nod_1 = ord_1*n_ele_1+1
    n_ele_2 = 6
    ord_2 = 1
    n_nod_2 = ord_2*n_ele_2+1
    global_coordinates = np.zeros((3,n_nod_1+n_nod_2))
    global_coordinates[0,:n_nod_1] = np.linspace(0,10,n_nod_1)
    global_coordinates[2,:n_nod_1] = 1.0
    global_coordinates[0,n_nod_1:n_nod_1+n_nod_2] = np.linspace(-1,11,n_nod_2)
    n_dim = global_coordinates.shape[0]
    n_nodes = global_coordinates.shape[1]
    n_dof = 7
    
    print("Constructing a vector of unknowns...")
    # vector of unknowns for the solver A.x = b
    x = np.zeros(shape=(n_dof,n_nodes))
    
    print("Constructing matrices of displacements, " + 
          "velocities and accelerations...")
    global_displacements = np.zeros((3,n_dim,n_nodes))
    global_velocities = np.zeros((3,n_dim,n_nodes))
    global_accelerations = np.zeros((3,n_dim,n_nodes))
    global_lagrange = np.zeros((3,n_nodes))

    history = []

    print("Numbering elements...")
    def ele_nodes(ele_id, n_nodes_per_ele):
        return np.array([
            n_nodes_per_ele*ele_id+j for j in range(n_nodes_per_ele+1)
        ], dtype=int)
    
    beam_1 = [
        el.SimoBeam(
            index=i,
            nodes=ele_nodes(i, ord_1),
            n_nodes_in_mesh=n_nodes,
            mesh_dof_per_node=n_dof,
            ref_vec=np.array([0,0,1]),
            coordinates=global_coordinates[:,ele_nodes(i, ord_1)],
            area=10.0,
            elastic_modulus=10.0,
            shear_modulus=10.0,
            inertia_primary=20.0,
            inertia_secondary=10.0,
            inertia_torsion=10.0,
            density=10.0,
            contact_radius=0.1
        ) for i in range(n_ele_1)
    ]
    beam_2 = [
        el.SimoBeam(
            index=i+n_ele_1,
            nodes=ele_nodes(i, ord_2)+n_nod_1,
            n_nodes_in_mesh=n_nodes,
            mesh_dof_per_node=n_dof,
            ref_vec=np.array([0,0,1]),
            coordinates=global_coordinates[:,ele_nodes(i, ord_2)+n_nod_1],
            area=10.0,
            elastic_modulus=10.0,
            shear_modulus=10.0,
            inertia_primary=20.0,
            inertia_secondary=10.0,
            inertia_torsion=10.0,
            density=10.0,
            contact_radius=0.1
        ) for i in range(n_ele_2)
    ]

    elements = [None, None, beam_1 + beam_2]
    
    contact_pairs = [  # (mortar side : non-mortar side)
        [np.arange(n_ele_1), np.arange(n_ele_2)+n_ele_1]
    ]
    for contact_pair in contact_pairs:
        mortar_side = [elements[2][i] for i in contact_pair[0]]
        mornod = ct.collect_nodes(mortar_side)
        mortar_contacts = []
        for i in contact_pair[1]: 
            mort = el.MortarContact(
                index=i+n_ele_2,
                parent_element=elements[2][i],
                mesh_dof_per_node=n_dof,
                nodal_locations_in_mesh=global_coordinates,
                mortar_nodes=mornod,
                mortar_elements=mortar_side
            )
            mortar_contacts.append(mort)
        elements[2] += mortar_contacts

    print("Identifing individual entities...")
    beams = ct.identify_entities(elements[2])
    print(len(beams), "beams found.")

    print("Determining the active degrees of freedom...")
    # Variable doesn't change with Newton iterations, only with time 
    #  step.
    active_dof = np.ones((2,n_dof,n_nodes), dtype=np.bool)
    active_dof[1][6] = False
    active_dof[1][(0,1,2,3,5),0] = False
    active_dof[1][(0,1,2,3,5),n_nod_1-1] = False
    active_dof[1][(0,1,2,3,5),n_nod_1] = False
    active_dof[1][(0,1,2,3,5),-1] = False
    
    print("Add force and/or displacements loads...")
    def Qload(t):
        Q = np.zeros(shape=(6, n_nodes))
        Qmax = -np.pi/3
        if t <= 15:
            Q[4,0] = -Qmax * t
            Q[4,n_nod_1-1] = Qmax * t
            Q[4,n_nod_1] = Qmax * t
            Q[4,-1] = -Qmax * t
        else:
            Q[4,0] = -Qmax
            Q[4,n_nod_1-1] = Qmax
            Q[4,n_nod_1] = Qmax
            Q[4,-1] = -Qmax
        return Q
    Qfollow = lambda t : np.zeros_like(Qload)
    Uload = lambda t : np.zeros(shape=(6, n_nodes))
    
    print("Note that additional nodal values can be added if necessary "
          + "simply by creating new matrices.")

    print("Initiating time...")
    # Variable doesn't change with Newton iterations, only with time 
    #  step.
    time_step = [None, 1.0]
    time = [None, 0.]
    final_time = 20.
    
    print("Selecting solver parameters...")
    max_number_of_time_steps = 300 #  100
    max_number_of_newton_iterations = 30
    max_number_of_contact_iterations = 10
    tolerance = 1e-8
    conv_test = "RES"
    # conv_test = "DSP"
    # conv_test = "ENE"
    
    print("We use Newmark-beta method.")
    beta = 0.25
    gamma = 0.5
    def matrix_multipliers():

        # if static
        c = (1.0, 0.0, 0.0)

        # if dynamic
        # c = (1.0, gamma/(time_step[1]*beta), 1/(time_step[1]**2*beta))
        
        return c
    
    print("Start of time loop...")
    for n in range(max_number_of_time_steps):
        print("Step", n)
        print("\tTime", time[1])

        ################################################################
        # Initiate time step
        time_step[0] = time_step[1]
        time[0] = time[1]

        c = matrix_multipliers()
        time[1] = time[0] + time_step[1]

        elements[0] = elements[2]
        global_displacements[0] = global_displacements[2]
        history.append(global_coordinates+global_displacements[0])
        global_velocities[0] = global_velocities[2]
        global_accelerations[0] = global_accelerations[2]
        global_lagrange[0] = global_lagrange[2]
        active_dof[0] = active_dof[1]
        
        # Apply displacement load
        x[:6] = Uload(time[1])
        ################################################################
        # Contact search
        contact_set_changed = True
        active_nodes_changed = True

        n_contact = 0
        while contact_set_changed or active_nodes_changed:
            contact_set_changed = False
            active_nodes_changed = False

            ############################################################
            # Newton-Raphson iteration
            for i in range(max_number_of_newton_iterations):
                # ------------------------------------------------------
                # Initiate new iteration
                tangent = np.zeros((n_dof*n_nodes, n_dof*n_nodes))

                elements[1] = elements[2]
                global_displacements[1] = global_displacements[2]
                global_velocities[1] = global_velocities[2]
                global_accelerations[1] = global_accelerations[2]
                global_lagrange[1] = global_lagrange[2]

                if i > 0:
                    # --------------------------------------------------
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
                            
                    # --------------------------------------------------        
                    # Solve system of equations.
                    mask = ~active_dof[1].flatten('F')
                    tangent[mask] = np.identity(n_dof*n_nodes)[mask]
                    tangent[:,mask] = np.identity(n_dof*n_nodes)[:,mask]
                    x = x.flatten('F')
                    x[mask] = np.zeros(shape=(n_dof*n_nodes))[mask]
                    np.savetxt('tang.csv', tangent)
                    x = np.linalg.solve(tangent, x)
                    x = np.reshape(x, newshape=(n_dof,n_nodes), order='F')
                    
                # ------------------------------------------------------
                # Update beam values
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
                
                # Update contact values
                global_lagrange[2] += x[6]

                # ------------------------------------------------------
                # Displacement convergence
                if conv_test == "DSP" and np.linalg.norm(x) <= tolerance:
                    print("Time step converged within", i+1, "iterations.")
                    break
                
                # ------------------------------------------------------
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
                
                # ------------------------------------------------------
                # Residual convergence
                res_norm = np.linalg.norm(x[active_dof[1]])
                if conv_test == "RES" and res_norm <= tolerance:
                    print(
                        "\tTime step converged within",
                        i+1, "iterations.\n"
                    )
                    break

            else:
                print("\nMaximum number of iterations reached without "
                    + "convergence!")
                return
        
            ############################################################
            # Continue contact
            # Newton-Raphson method converged to a new solution
            # Contact search
            for contact_pair in contact_pairs:
                contact_pair[0] = contact_pair[0]
                contact_pair[1] = contact_pair[1]
            contact_set_changed = False
            
            # Partner mortar nodes
            for contact_pair in contact_pairs:
                mortar_side = [elements[2][i] for i in contact_pair[0]]
                mornod = ct.collect_nodes(mortar_side)
                for nmb in contact_pair[1]:
                    for e in elements[2]:
                        if type(e) == el.MortarContact:
                            if e.parent.index == nmb:
                                mort = e
                    old_parent = mort.parent.index
                    mort.find_partners(
                        global_coordinates+global_displacements[2],
                        mornod,
                        mortar_side
                    )
                    if mort.parent.index != old_parent:
                        active_nodes_changed = True

            # Non-mortar node activation
            for e in elements[2]:
                if type(e) == el.MortarContact:
                    act = e.activated_nodes(
                        global_lagrange[2][e.nodes],
                        active_dof[1][6,e.nodes]
                    )
                    current_status = active_dof[1][6,e.nodes]
                    if np.all(act == current_status):
                        continue
                    else:
                        active_dof[1][6,e.nodes] = act
                        active_nodes_changed = True
            
            n_contact += 1
            if n_contact > max_number_of_contact_iterations:
                print("\nContact: Maximum number of contact iterations",
                      "reached without convergence!")
                return

        ################################################################
        # Finish time step
        if time[1] >= final_time:
            print("Computation is finished, reached the end of time.")
            history.append(global_coordinates+global_displacements[0].copy())
            h = np.array(history)
            return h

    print("Final time was never reached.")
    return

def main():
    h = patch()
    n_nod_1 = 7
    if type(h) == np.ndarray:
        for i in range(0,len(h),2):
            plt.plot(h[i,0,:n_nod_1],h[i,2,:n_nod_1], '.-', label=i)
            plt.plot(h[i,0,n_nod_1:],h[i,2,n_nod_1:], '.-')
        plt.xlim((-2,13))
        plt.ylim((-3,7))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
