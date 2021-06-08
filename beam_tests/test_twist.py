import os
import sys

cwd = os.getcwd()
folder = os.path.basename(cwd)
while folder != "beam":
    cwd = os.path.dirname(cwd)
    folder = os.path.basename(cwd)
    if len(cwd) == 0:
        print("Root directory was not found. Try inserting the path manually with 'sys.path.insert(0, absolute_path_to_root)'")
        sys.exit()
print("Root directory:", cwd)
sys.path.insert(0, cwd)

import functools
import numpy as np
from system import System
import mesh
import postprocessing as postproc
import matplotlib.pyplot as plt


def case1(nele):
    """
    In this example, one cantilever beam is bent towards another.
    Contact, static analysis.
    """
    
    mat = {
        'EA':np.pi*1.0e2,
        'GA1':1.2083e2,
        'GA2':1.2083e2,
        'GIt':6.03846,
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
        mesh.add_mortar_element([ele], possible_contact_partners=pce)

    coordinates = np.hstack((coordinates1, coordinates2))
    elements = elements1 + elements2
    system = System(coordinates, elements)
    system.time_step = 1.0
    system.max_number_of_time_steps = 1000
    system.max_number_of_contact_iterations = 30
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
    Contact, static analysis.
    """
    
    mat = {
        'EA':np.pi*1.0e2,
        'GA1':1.2083e2,
        'GA2':1.2083e2,
        'GIt':6.03846,
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
    system11 = case2(10)
    system11.solve()
    system12 = case2(20)
    system12.solve()
    system13 = case2(40)
    system13.solve()
    L = 5.0
    d = 0.02
    postproc.contact_force_plot(system11, -1, savefig=True, color='tab:blue')
    postproc.contact_force_plot(system12, -1, savefig=True, color='tab:orange')
    postproc.contact_force_plot(system13, -1, savefig=True, color='tab:green')
    plt.close()
    # for i in range(0, len(system.time), 1):
    #     postproc.line_plot(system, (-d,L+d), (-L/20-d,L/20+d), (-L/20-d,L/20+d), i, include_initial_state=False, savefig=False, camera=None)
    #     postproc.gap_plot(system, i)
    #     postproc.contact_force_plot(system, i, savefig=False)
    # postproc.line_plot(system, (-d,L+d), (-L/20-d,L/20+d), (-L/20-d,L/20+d), -1, include_initial_state=False, savefig=True, camera=None)
    # postproc.gap_plot(system, -1, savefig=True)
    # postproc.contact_force_plot(system, -1, savefig=True)

if __name__ == "__main__":
    main()
