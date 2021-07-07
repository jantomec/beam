import functools
import numpy as np
from beam.system import System
from beam import mesh
from beam import postprocessing as postproc
import matplotlib.pyplot as plt


def case1(nele, nint):
    """
    In this example, one cantilever beam is bent towards another.
    Beams are initially touching.
    Contact, static analysis.
    """
    
    mat = {
        'EA':np.pi*1.0e5,
        'GA1':1.2083e5,
        'GA2':1.2083e5,
        'GIt':6.04158,
        'EI1':7.85398,
        'EI2':7.85398,
        'Arho':1.0,
        'I12rho':1.0,
        'I1rho':1.0,
        'I2rho':1.0,
        'Contact radius':0.01
    }
    
    (coordinates1, elements1) = mesh.line_mesh(A=(0,0,0.01), B=(5,0,0.01), n_elements=nele, order=2, material=mat, reference_vector=(0,0,1))
    (coordinates2, elements2) = mesh.line_mesh(A=(0,0,-0.01), B=(5,0,-0.01), n_elements=nele, order=2, material=mat, reference_vector=(0,0,1),
                                               starting_node_index=coordinates1.shape[1])
   
    neighbouring_elements = 1
    for (i2, ele) in enumerate(elements2):
        if i2 == 0:
            pce = elements1[:neighbouring_elements+1]
        else:
            pce = elements1[i2-neighbouring_elements:i2+neighbouring_elements]
        mesh.add_mortar_element([ele], possible_contact_partners=pce, n_contact_integration_points=nint)

    coordinates = np.hstack((coordinates1, coordinates2))
    elements = elements1 + elements2
    system = System(coordinates, elements)
    system.time_step = 1.0
    system.max_number_of_time_steps = 1000
    system.max_number_of_contact_iterations = 50
    system.final_time = 9.0
    system.solver_type = 'static'
    system.tolerance = 1e-6
    system.convergence_test_type = 'RES'
    system.contact_detection = True
    system.print_residual = True
    
    def user_displacement_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        radius = self.coordinates[2,0]
        freq = 2*np.pi/8
        if self.current_time > 1:
            t = self.current_time-1
            Q[1,coordinates1.shape[1]-1] = 0.01*(np.sin(freq*t) - np.sin(freq*(t - self.time_step)))
            Q[2,coordinates1.shape[1]-1] = 0.01*(np.cos(freq*t) - np.cos(freq*(t - self.time_step)))
            Q[1,-1] = -0.01*(np.sin(freq*t) - np.sin(freq*(t - self.time_step)))
            Q[2,-1] = -0.01*(np.cos(freq*t) - np.cos(freq*(t - self.time_step)))
        else:
            Q[0,coordinates1.shape[1]-1] = 0.04967
            Q[0,-1] = 0.04967
        return Q

    
    system.degrees_of_freedom[-1][:6,0] = False  # [current time, dof 0 through 5, first node of the first beam]
    system.degrees_of_freedom[-1][:6,coordinates1.shape[1]] = False  # [current time, dof 0 through 5, first node of the second beam]
    system.degrees_of_freedom[-1][:6, coordinates1.shape[1]-1] = False  # [current time, dof 0 through 5, last node of the first beam]
    system.degrees_of_freedom[-1][:6,-1] = False  # [current time, dof 0 through 5, last node of the second beam]
    system.displacement_load = functools.partial(user_displacement_load, system)
    
    return system

def case2(nele):
    """
    In this example, one cantilever beam is bent towards another.
    Beams are initially with some gap between them.
    Contact, static analysis.
    """
    
    mat = {
        'EA':np.pi*1.0e5,
        'GA1':1.2083e2,
        'GA2':1.2083e2,
        'GIt':6.04158,
        'EI1':7.85398,
        'EI2':7.85398,
        'Arho':1.0,
        'I12rho':1.0,
        'I1rho':1.0,
        'I2rho':1.0,
        'Contact radius':0.01
    }
    
    (coordinates1, elements1) = mesh.line_mesh(A=(0,0,0.02), B=(5,0,0.02), n_elements=nele, order=3, material=mat, reference_vector=(0,0,1))
    (coordinates2, elements2) = mesh.line_mesh(A=(0,0,-0.02), B=(5,0,-0.02), n_elements=nele, order=3, material=mat, reference_vector=(0,0,1),
                                               starting_node_index=coordinates1.shape[1])
   
    neighbouring_elements = 1
    for (i2, ele) in enumerate(elements2):
        if i2 == 0:
            pce = elements1[:neighbouring_elements+1]
        else:
            pce = elements1[i2-neighbouring_elements:i2+neighbouring_elements]
        mesh.add_mortar_element([ele], possible_contact_partners=pce, n_contact_integration_points=5)

    coordinates = np.hstack((coordinates1, coordinates2))
    elements = elements1 + elements2
    system = System(coordinates, elements)
    system.time_step = 1.0
    system.max_number_of_time_steps = 1000
    system.max_number_of_contact_iterations = 10
    system.final_time = 9.0
    system.solver_type = 'static'
    system.convergence_test_type = 'RES'
    system.contact_detection = True
    system.print_residual = True
    
    def user_displacement_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        radius = self.coordinates[2,0]
        freq = 2*np.pi/8
        if self.current_time > 1:
            t = self.current_time-1
            Q[1,coordinates1.shape[1]-1] = 0.01*(np.sin(freq*t) - np.sin(freq*(t - self.time_step)))
            Q[2,coordinates1.shape[1]-1] = 0.01*(np.cos(freq*t) - np.cos(freq*(t - self.time_step)))
            Q[1,-1] = -0.01*(np.sin(freq*t) - np.sin(freq*(t - self.time_step)))
            Q[2,-1] = -0.01*(np.cos(freq*t) - np.cos(freq*(t - self.time_step)))
        else:
            Q[2,coordinates1.shape[1]-1] = -0.01
            Q[2,-1] = 0.01
        return Q

    
    system.degrees_of_freedom[-1][:6,0] = False  # [current time, dof 0 through 5, first node of the first beam]
    system.degrees_of_freedom[-1][:6,coordinates1.shape[1]] = False  # [current time, dof 0 through 5, first node of the second beam]
    system.degrees_of_freedom[-1][:6, coordinates1.shape[1]-1] = False  # [current time, dof 0 through 5, last node of the first beam]
    system.degrees_of_freedom[-1][:6,-1] = False  # [current time, dof 0 through 5, last node of the second beam]
    system.displacement_load = functools.partial(user_displacement_load, system)
    
    return system

def main():
    np.set_printoptions(linewidth=10000, edgeitems=2000)
    system11 = case1(8, 8)
    system11.solve()
    system12 = case1(16, 8)
    system12.solve()
    # system13 = case1(32, 8)
    # system13.solve()
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 20
    })
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    f1 = system11.lagrange[-1][len(system11.lagrange[-1])//2:]
    f2 = system12.lagrange[-1][len(system12.lagrange[-1])//2:]
    # f3 = system13.lagrange[-1][len(system13.lagrange[-1])//2:]
    x1 = np.linspace(0, 5, len(f1))
    x2 = np.linspace(0, 5, len(f2))
    # x3 = np.linspace(0, 5, len(f3))
    ax.plot(x1, f1, ".", label='8 elements')
    ax.plot(x2, f2, "^", label='16 elements')
    # ax.plot(x3, f3, "s", label='32 elements')
    ax.set_xlim((x1[0], x1[-1]))
    ax.set_xlabel('s')
    ax.set_ylabel('$\lambda$')
    ax.legend()
    #plt.savefig("img2.pdf", bbox_inches='tight')
    plt.show()

    # L = 5.0
    # d = 0.02
    # postproc.line_plot(system11, (-d,L+d), (-L/20-d,L/20+d), (-L/20-d,L/20+d), -1, include_initial_state=False, savefig=False, camera=None)
    # raw_data = postproc.line_plot_raw_data(system11, -1)
    # np.savetxt('data1.txt', raw_data[0])
    # np.savetxt('data2.txt', raw_data[1])

if __name__ == "__main__":
    main()
