import numpy as np
import matplotlib.pyplot as plt
import contact

def line_plot(system, xlim, ylim, zlim, time_step, include_initial_state=True, savefig=False, camera=None):
    i = time_step
    beams_id = contact.identify_entities(system.elements)
    beams = []
    for beam_id in beams_id:
        beam = []
        for e_id in beam_id:
            beam.append(system.elements[e_id])
        beams.append(beam)
    
    color_map = plt.get_cmap("tab10")
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Time ' + str(np.round(system.time[i], 3)))
    
    n_plot_points_for_each_element = 20

    for (b, beam) in enumerate(beams):
        nodes = contact.collect_nodes(beam)

        x0 = system.coordinates[0][nodes]
        y0 = system.coordinates[1][nodes]
        z0 = system.coordinates[2][nodes]
        if include_initial_state:
            ax.plot3D(x0, y0, z0, '.', color=color_map(b))

        u0 = x0 + system.displacement[i][0][nodes]
        v0 = y0 + system.displacement[i][1][nodes]
        w0 = z0 + system.displacement[i][2][nodes]
        ax.plot3D(u0, v0, w0, '.', color=color_map(b))

        boundary_conditions = []
        for j in range(system.degrees_of_freedom[i][:,nodes].shape[1]):
            if np.any(~system.degrees_of_freedom[i][:,nodes][:6,j]):
                boundary_conditions.append(j)
        ax.plot3D(u0[boundary_conditions], v0[boundary_conditions], w0[boundary_conditions], '.', color='black')

        for ele in beam:
            N = ele.Ndis[0](np.linspace(-1,1,n_plot_points_for_each_element))
            x = system.coordinates[:,ele.nodes] @ N
            if include_initial_state:
                ax.plot3D(x[0], x[1], x[2], '--', color=color_map(b), alpha=0.5)
            u = x + system.displacement[i][:,ele.nodes] @ N
            ax.plot3D(u[0], u[1], u[2], '-', color=color_map(b), alpha=0.5)
            # ax.plot3D(u[0], u[1], u[2], '-', linewidth=6.0, color=color_map(b), alpha=0.5)
    
    if camera is not None:
        ax.view_init(elev=camera[1], azim=camera[0])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if savefig:
        plt.savefig('image'+str(time_step)+'.png')
    else:
        plt.show()

def gap_plot(system, time_step, savefig=False):
    i = time_step
    plt.title('Time ' + str(np.round(system.time[i], 3)))
    plt.plot(system.gap_function[i][:,0], system.gap_function[i][:,1], '.')
    plt.xlabel('s')
    plt.ylabel('gap')
    plt.hlines(0, xmin=plt.xlim()[0], xmax=plt.xlim()[1])
    if savefig:
        plt.savefig('gap_function-'+str(time_step)+'.png')
    else:
        plt.show()

def contact_force_plot(system, time_step, savefig=False, color=None):
    """
    Return gap function values along the centreline.
    """
    i = time_step
    f = []
    x0 = 0
    for ele in system.elements:
        try:
            contact_element = ele.child
            sdom = np.linspace(-1,1)
            for s in sdom:
                x = x0 + (s+1) * contact_element.parent.jacobian
                Phi1 = contact_element.Nlam[0](s)
                lam = system.lagrange[i][contact_element.parent.nodes] @ Phi1
                f.append([x,lam])
            x0 += contact_element.parent.jacobian * 2
        except AttributeError:
            continue
    f = np.array(f)

    plt.title('Time ' + str(np.round(system.time[i], 3)))
    if color is None:
        plt.plot(f[:,0], f[:,1], '-')
    else:
        plt.plot(f[:,0], f[:,1], '-', color=color, label=str(int(system.get_number_of_elements()/2))+" elements")
        plt.legend()
    plt.xlabel('s')
    plt.xlim((f[0,0], f[-1,0]))
    plt.ylabel('contact force')
    plt.hlines(0, xmin=plt.xlim()[0], xmax=plt.xlim()[1])
    if savefig:
        plt.savefig('force_function-'+str(time_step)+'.png')
    else:
        plt.show()

def energy_plot(system):
    t = np.array(system.time)
    ek = np.array(system.kinetic_energy)
    ep = np.array(system.potential_energy)
    e = ek + ep
    fig = plt.figure()
    ax = plt.axes()
    ax.set_title('Energy plot')
    ax.plot(t, ek, '.-', label='Kinetic energy')
    ax.plot(t, ep, '.-', label='Potential energy')
    ax.plot(t, e, '.-', label='Total energy')
    plt.show()
