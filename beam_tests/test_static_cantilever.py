import os
import sys

cwd = os.getcwd()
folder = os.path.basename(cwd)
while folder != "beam":
    print("Hello")
    print(cwd, os.path.dirname(cwd))
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
    In this example, a cantilever beam is bent into a double circle.
    No contact, static analysis.
    """
    
    mat = {
        'area':1.0,
        'elastic_modulus':1.0,
        'shear_modulus':1.0,
        'inertia_primary':2.0,
        'inertia_secondary':1.0,
        'inertia_torsion':1.0,
        'density':1.0,
        'contact_radius':1.0
    }
    
    (coordinates, elements) = mesh.line_mesh(A=(0,0,0), B=(1,0,0), n_elements=5, order=1, material=mat, reference_vector=(0,0,1))
    system = System(coordinates, elements)
    system.time_step = 1.0
    system.final_time = 1.0
    system.solver_type = 'static'
    system.contact_detection=False
    
    def user_force_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        Q[4,-1] = 8*np.pi
        return Q
    
    system.degrees_of_freedom[-1][:6,0] = False  # [current time, dof 0 through 5, node 0]
    
    system.force_load = functools.partial(user_force_load, system)
    
    return system

def main():
    system = case()
    system.solve()
    
    for i in range(len(system.time)):
        postproc.line_plot(system, (-0.2,1.2), (-0.7,0.7), (-0.7,0.7), i)

if __name__ == "__main__":
    main()
