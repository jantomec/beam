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
        'elastic_modulus':100,
        'shear_modulus':10,
        'inertia_primary':100.785398,
        'inertia_secondary':100.785398,
        'inertia_torsion':1.5708,
        'density':1,
        'contact_radius':2
    }
    
    (coordinates1, elements1) = mesh.line_mesh(A=(0,0,6), B=(50,0,6), n_elements=1, order=1, material=mat, reference_vector=(0,0,1))
    (coordinates2, elements2) = mesh.line_mesh(A=(0,0,-6), B=(50,0,-6), n_elements=1, order=1, material=mat, reference_vector=(0,0,1),
                                               starting_node_index=coordinates1.shape[1], possible_contact_partners=elements1,
                                               dual_basis_functions=True, n_contact_integration_points=10)
    
    coordinates = np.hstack((coordinates1, coordinates2))
    elements = elements1 + elements2
    system = System(coordinates, elements)
    system.time_step = 1.0
    system.final_time = 1.0
    system.solver_type = 'static'
    system.contact_detection = True
    system.print_residual = True
    system.max_number_of_newton_iterations = 60
    
    def user_force_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        Q[2,-1] = 8.0 * self.current_time / self.final_time
        return Q
    
    system.degrees_of_freedom[-1][:6,:coordinates1.shape[1]] = False  # [current time, dof 0 through 5, first node of the first beam]
    system.degrees_of_freedom[-1][:6,coordinates1.shape[1]] = False  # [current time, dof 0 through 5, all nodes of the second beam]
    system.force_load = functools.partial(user_force_load, system)
    
    return system

def main():
    np.set_printoptions(linewidth=1000, edgeitems=1000, precision=3, suppress=True)

    system = case()
    system.solve()
    
    for i in range(len(system.time)):
        postproc.line_plot(system, (-2,52), (-7,7), (-7,7), i)

if __name__ == "__main__":
    main()
