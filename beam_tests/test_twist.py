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


def case():
    """
    In this example, one cantilever beam is bent towards another.
    Contact, static analysis.
    """
    
    mat = {
        'area':np.pi,
        'elastic_modulus':2.1e5,
        'shear_modulus':8.0e4,
        'inertia_primary':0.785398,
        'inertia_secondary':0.785398,
        'inertia_torsion':1.5708,
        'density':8.0e-7,
        'contact_radius':2.0
    }
    
    (coordinates1, elements1) = mesh.line_mesh(A=(0,0,4), B=(500,0,4), n_elements=4, order=2, material=mat, reference_vector=(0,0,1), n_contact_integration_points=10)
    (coordinates2, elements2) = mesh.line_mesh(A=(0,0,-4), B=(500,0,-4), n_elements=4, order=2, material=mat, reference_vector=(0,0,1),
                                               starting_node_index=coordinates1.shape[1], possible_contact_partners=elements1, dual_basis_functions=False)
    
    coordinates = np.hstack((coordinates1, coordinates2))
    elements = elements1 + elements2
    system = System(coordinates, elements)
    system.time_step = 0.2
    system.max_number_of_time_steps = 500
    system.max_number_of_contact_iterations = 4
    system.final_time = 2.0
    system.solver_type = 'static'
    system.convergence_test_type = 'RES'
    system.contact_detection = True
    system.print_residual = True
    
    def user_displacement_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        freq = 5.0
        radius = 4.0
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

def main():
    system = case()
    system.solve()
    
    #for i in range(0, len(system.time), 20):
    #    L = system.coordinates[0,-1]
    #    d = 10
    #    postproc.line_plot(system, (-d,L+d), (-L/20-d,L/20+d), (-L/20-d,L/20+d), i, include_initial_state=False)

    L = system.coordinates[0,-1]
    d = 10
    postproc.line_plot(system, (-d,L+d), (-L/20-d,L/20+d), (-L/20-d,L/20+d), -1, include_initial_state=False)


if __name__ == "__main__":
    main()
