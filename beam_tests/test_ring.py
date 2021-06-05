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
    In this example, a ring is twisted. A ring is clamped at some point
    and a twisting moment is applied at the opposite point.
    Contact, static analysis.
    """
    
    mat = {
        'EA':2.76461e3,
        'GA1':1.03924e3,
        'GA2':1.03924e3,
        'GIt':2078.5,
        'EI1':2764.52,
        'EI2':2764.52,
        'Arho':1.0,
        'I12rho':1.0,
        'I1rho':1.0,
        'I2rho':1.0,
        'Contact radius':0.02 * 2 * np.pi
    }

    (coordinates, elements) = mesh.circle_mesh(
        R=1, n_elements=11, order=2, material=mat,
        reference_vector=np.array([0,0,1]), plane=(0,1),
        starting_node_index=0
    )

    exclude_range = 2

    for i in range(len(elements)):
        mortar_elements = elements.copy()
        exclude_indices = []
        for j in range(-exclude_range, exclude_range+1):
            exclude_indices.append((i+j)%len(elements))
        exclude_indices.sort(reverse=True)
        for j in exclude_indices:
            mortar_elements.pop(j)

        mesh.add_mortar_element(
            [elements[i]],
            possible_contact_partners=mortar_elements.copy(),
            n_contact_integration_points=10
        )

    system = System(coordinates, elements)
    system.time_step = 1.0
    system.max_number_of_time_steps = 1000
    system.max_number_of_contact_iterations = 10
    system.final_time = 14.0
    system.solver_type = 'static'
    system.convergence_test_type = 'RES'
    system.contact_detection = True
    system.print_residual = True
    system.tolerance = 1e-6
    
    def user_force_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        if self.current_time < 9:
            Q[3,n_nodes//2] = 700*self.current_time
        else:
            Q[3,n_nodes//2] = 700*9 + 70*(self.current_time - 9)
        return Q
    
    system.degrees_of_freedom[-1][:6,0] = False  # [current time, dof 0 through 5, first node of the first beam]
    system.degrees_of_freedom[-1][6,:] = False
    system.force_load = functools.partial(user_force_load, system)
    
    return system

def main():
    np.set_printoptions(linewidth=10000, edgeitems=2000)
    system = case()
    system.solve()
    
    limits = (-1.6, 1.6)
    for i in range(0, len(system.time)):
       postproc.line_plot(system, limits, limits, limits, i, include_initial_state=False, savefig=False, camera=(12.0, 33.0))

if __name__ == "__main__":
    main()
