import numpy as np
import matplotlib.pyplot as plt
import contact

def line_plot(system, xlim, ylim, zlim, time_step, include_initial_state=True, savefig=False):
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
    ax.set_title('Time ' + str(system.time[i]))
    
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

        for ele in beam:
            N = ele.Ndis[0](np.linspace(-1,1,n_plot_points_for_each_element))
            x = system.coordinates[:,ele.nodes] @ N
            if include_initial_state:
                ax.plot3D(x[0], x[1], x[2], '--', color=color_map(b), alpha=0.5)
            u = x + system.displacement[i][:,ele.nodes] @ N
            ax.plot3D(u[0], u[1], u[2], '-', color=color_map(b), alpha=0.5)
            # ax.plot3D(u[0], u[1], u[2], '-', linewidth=6.0, color=color_map(b), alpha=0.5)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    if savefig:
        plt.savefig('image'+str(time_step)+'.png')
    else:
        plt.show()

def gap_plot(system):
    return

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
