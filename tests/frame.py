import functools
import numpy as np
from beam.system import System
from beam import mesh
from beam import postprocessing as postproc


def case():
    """
    In this example, a right angle frame is bent by an out-of-plane dynamic force.
    No contact, dynamic.
    """
    
    mat = {
        'Area':1000.0,
        'Elastic modulus':1000.0,
        'Shear modulus':1000.0,
        'Inertia primary':1.0,
        'Inertia secondary':1.0,
        'Inertia torsion':1.0,
        'Shear coefficient primary':1.0,
        'Shear coefficient secondary':1.0,
        'Density':10.0,
        'Contact radius':1.0,
        'Arho':1.0  # Arho overrides A*rho
    }
    
    points = np.array([[0, 0, 10],
                       [0, 0, 0],
                       [0, 10, 10]], dtype=float)
    (coordinates, elements) = mesh.n_point_mesh(points, n_elements=[1,1], order=2, material=mat, reference_vector=(0,1,1))
    
    # hard-code Arho for this specific example
    for ele in elements:
        ele.prop.Arho = 1.0

    system = System(coordinates, elements)
    system.time_step = 0.25
    system.final_time = 2.0
    system.solver_type = 'dynamic'
    system.max_number_of_time_steps = 250
    system.contact_detection=True
    
    def user_force_load(self):
        n_nodes = self.get_number_of_nodes()
        Q = np.zeros((6, n_nodes))
        if self.current_time <= 1:
            Q[1,2] = 50 * self.current_time
        elif 1 < self.current_time and self.current_time < 2:
            Q[1,2] = 50 - 50 * (self.current_time - 1)
        return Q

    system.degrees_of_freedom[-1][:6,0] = False  # [current time, dof 0 through 5, node 0]
    system.force_load = functools.partial(user_force_load, system)

    return system

def main():
    frame = case()
    frame.solve()
    for i in range(len(frame.time)):
        postproc.line_plot(frame, (-2,12), (-7,7), (-2,12), i)

if __name__ == "__main__":
    main()
