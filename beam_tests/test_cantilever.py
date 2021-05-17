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

def case2():
    """
    In this example, a beam fixed on both ends while the right end is being moved upward via displacement control.
    No contact, static analysis.
    """
    
    mat = {
        'area':20.0,
        'elastic_modulus':1.0,
        'shear_modulus':10.0,
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
    system.convergence_test_type = 'DSP'

    def user_displacement_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        Q[2,-1] = 0.5
        return Q
    
    system.degrees_of_freedom[-1][:6,0] = False  # [current time, dof 0 through 5, node 0]
    system.degrees_of_freedom[-1][:6,-1] = False  # [current time, dof 0 through 5 , last node]

    system.displacement_load = functools.partial(user_displacement_load, system)
    
    return system

def case3():
    """
    In this example, one cantilever beam is bent towards another. Only one linear element for each beam.
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

def case4():
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
        'contact_radius':1.0
    }
    
    (coordinates1, elements1) = mesh.line_mesh(A=(0,0,2), B=(50,0,2), n_elements=4, order=1, material=mat, reference_vector=(0,0,1))
    (coordinates2, elements2) = mesh.line_mesh(A=(0,0,-2), B=(50,0,-2), n_elements=4, order=1, material=mat, reference_vector=(0,0,1),
                                               starting_node_index=coordinates1.shape[1], possible_contact_partners=elements1)
    
    coordinates = np.hstack((coordinates1, coordinates2))
    elements = elements1 + elements2
    system = System(coordinates, elements)
    system.time_step = 1.0
    system.final_time = 1.0
    system.solver_type = 'static'
    system.convergence_test_type = 'RES'
    system.contact_detection = True
    system.print_residual = True
    
    def user_force_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        Q[2,coordinates1.shape[1]-2] = -15 * self.current_time / self.final_time
        return Q
    
    system.degrees_of_freedom[-1][:6,0] = False  # [current time, dof 0 through 5, first node of the first beam]
    system.degrees_of_freedom[-1][:6,coordinates1.shape[1]] = False  # [current time, dof 0 through 5, first node of the second beam]
    system.force_load = functools.partial(user_force_load, system)
    
    return system


def main():
    system = case1()
    system.solve()
    
    for i in range(len(system.time)):
        postproc.line_plot(system, (-0.2,1.2), (-0.7,0.7), (-0.7,0.7), i)


if __name__ == "__main__":
    main()
