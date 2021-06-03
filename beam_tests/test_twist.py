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


def case1():
    """
    In this example, one cantilever beam is bent towards another.
    Contact, static analysis.
    """
    
    mat = {
        'area':0.000314159,
        'elastic_modulus':1.0e9,
        'shear_modulus':0.3846e9,
        'inertia_primary':7.85398e-9,
        'inertia_secondary':7.85398e-9,
        'inertia_torsion':1.5708e-8,
        'density':8.0e-7,
        'contact_radius':0.01
    }
    
    (coordinates1, elements1) = mesh.line_mesh(A=(0,0,0.02), B=(5,0,0.02), n_elements=4, order=2, material=mat, reference_vector=(0,0,1))
    (coordinates2, elements2) = mesh.line_mesh(A=(0,0,-0.02), B=(5,0,-0.02), n_elements=4, order=2, material=mat, reference_vector=(0,0,1),
                                               starting_node_index=coordinates1.shape[1], possible_contact_partners=elements1, dual_basis_functions=False)
    
    coordinates = np.hstack((coordinates1, coordinates2))
    elements = elements1 + elements2
    system = System(coordinates, elements)
    system.time_step = 0.1
    system.max_number_of_time_steps = 1000
    system.max_number_of_contact_iterations = 10
    system.final_time = 10.0
    system.solver_type = 'static'
    system.convergence_test_type = 'RES'
    system.contact_detection = True
    system.print_residual = True
    
    def user_displacement_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        freq = 1.0
        radius = 0.02
        if self.current_time > 0:
            Q[1,coordinates1.shape[1]-1] = radius*np.sin(freq*self.current_time) - radius*np.sin(freq*(self.current_time - self.time_step))
            Q[2,coordinates1.shape[1]-1] = radius*np.cos(freq*self.current_time) - radius*np.cos(freq*(self.current_time - self.time_step))
            Q[1,-1] = -radius*np.sin(freq*self.current_time) + radius*np.sin(freq*(self.current_time - self.time_step))
            Q[2,-1] = -radius*np.cos(freq*self.current_time) + radius*np.cos(freq*(self.current_time - self.time_step))
        return Q

    
    system.degrees_of_freedom[-1][:6,0] = False  # [current time, dof 0 through 5, first node of the first beam]
    system.degrees_of_freedom[-1][:6,coordinates1.shape[1]] = False  # [current time, dof 0 through 5, first node of the second beam]
    system.degrees_of_freedom[-1][:6, coordinates1.shape[1]-1] = False  # [current time, dof 0 through 5, last node of the first beam]
    system.degrees_of_freedom[-1][:6,-1] = False  # [current time, dof 0 through 5, last node of the second beam]
    system.displacement_load = functools.partial(user_displacement_load, system)
    
    return system

def case2():
    """
    Simulation of the static twisting process of a rope.
    Contact, static analysis.
    """
    
    mat = {
        'area':0.000314159,
        'elastic_modulus':1.0e9,
        'shear_modulus':0.3846e9,
        'inertia_primary':7.85398e-9,
        'inertia_secondary':7.85398e-9,
        'inertia_torsion':1.5708e-8,
        'density':8.0e-7,
        'contact_radius':0.01
    }
    
    (coordinates1, elements1) = mesh.line_mesh(A=(0,0,0.02), B=(5,0,0.02), n_elements=30, order=1, material=mat, reference_vector=(0,0,1))
    (coordinates2, elements2) = mesh.line_mesh(A=(0,0,-0.02), B=(5,0,-0.02), n_elements=30, order=1, material=mat, reference_vector=(0,0,1),
                                               starting_node_index=coordinates1.shape[1])
    
    mesh.add_mortar_element(elements2, possible_contact_partners=elements1, n_contact_integration_points=10)
    # mesh.add_mortar_element(elements1, possible_contact_partners=elements2)

    coordinates = np.hstack((coordinates1, coordinates2))
    elements = elements1 + elements2
    system = System(coordinates, elements)
    system.time_step = 1.0
    system.max_number_of_time_steps = 1000
    system.max_number_of_contact_iterations = 10
    system.max_number_of_newton_iterations = 30
    system.final_time = 80.0
    system.tolerance = 1e-6
    system.solver_type = 'static'
    system.convergence_test_type = 'RES'
    system.contact_detection = True
    system.print_residual = True
    
    def user_displacement_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        freq = 4*2*np.pi / system.final_time
        radius = self.coordinates[2,0]
        if self.current_time > 0:
            Q[1,coordinates1.shape[1]-1] = radius*np.sin(freq*self.current_time) - radius*np.sin(freq*(self.current_time - self.time_step))
            Q[2,coordinates1.shape[1]-1] = radius*np.cos(freq*self.current_time) - radius*np.cos(freq*(self.current_time - self.time_step))
            Q[1,-1] = -radius*np.sin(freq*self.current_time) + radius*np.sin(freq*(self.current_time - self.time_step))
            Q[2,-1] = -radius*np.cos(freq*self.current_time) + radius*np.cos(freq*(self.current_time - self.time_step))
        return Q

    
    system.degrees_of_freedom[-1][:6,0] = False  # [current time, dof 0 through 5, first node of the first beam]
    system.degrees_of_freedom[-1][:6,coordinates1.shape[1]] = False  # [current time, dof 0 through 5, first node of the second beam]
    system.degrees_of_freedom[-1][[0,1,2,4,5], coordinates1.shape[1]-1] = False  # [current time, dof 0 through 5, last node of the first beam]
    system.degrees_of_freedom[-1][[0,1,2,4,5],-1] = False  # [current time, dof 0 through 5, last node of the second beam]
    system.displacement_load = functools.partial(user_displacement_load, system)
    
    return system

def main():
    np.set_printoptions(linewidth=10000, edgeitems=2000)
    system = case2()
    system.solve()
    
    L = system.coordinates[0,-1]
    d = 0.02
    for i in range(0, len(system.time), 1):
       postproc.line_plot(system, (-d,L+d), (-L/20-d,L/20+d), (-L/20-d,L/20+d), i, include_initial_state=False)
    postproc.line_plot(system, (-d,L+d), (-L/20-d,L/20+d), (-L/20-d,L/20+d), -1, include_initial_state=False)

if __name__ == "__main__":
    main()
