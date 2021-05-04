#
#
#To do:
#    Change possible contact partners to contact search radius: elements within this radius excluding the element itself will be in possible contact partners. This should get computed every contact iteration.
    

import numpy as np
import elements as elmt


def __ele_nodes(ele_id, n_nodes_per_ele):
    return np.array([
        n_nodes_per_ele*ele_id+j for j in range(n_nodes_per_ele+1)
    ], dtype=int)

def line_mesh(A, B, n_elements, order, material, reference_vector, possible_contact_partners=[], consider_contact_jacobian=False, dual_basis_functions=True):
    """
    Create line mesh from coordinate A to B.
    """
    n_ele = n_elements
    n_nod = order * n_ele + 1
    coordinates = np.zeros((3,n_nod))
    for i in range(3):
        coordinates[i,:] = np.linspace(A[i], B[i], n_nod)
    
    beam = []
    for i in range(n_ele):
        element = elmt.SimoBeam(
            nodes=__ele_nodes(i, order),
            n_nodes_in_mesh=n_nod,
            mesh_dof_per_node=7,
            ref_vec=reference_vector,
            coordinates=coordinates[:,__ele_nodes(i, order)],
            area=material['area'],
            elastic_modulus=material['elastic_modulus'],
            shear_modulus=material['shear_modulus'],
            inertia_primary=material['inertia_primary'],
            inertia_secondary=material['inertia_secondary'],
            inertia_torsion=material['inertia_torsion'],
            density=material['density'],
            contact_radius=material['contact_radius']
        )
        if len(possible_contact_partners) != 0:
            element.child = elmt.MortarContact(
                parent_element=element,
                n_integration_points=n_nod,
                possible_contact_partners=possible_contact_partners,
                consider_jacobian=consider_contact_jacobian,
                dual_basis_functions=dual_basis_functions

            )
        beam.append(element)
    return (coordinates, beam)

def three_point_mesh(A, B, C, n_elements, order, material, reference_vector, possible_contact_partners=[], consider_contact_jacobian=False, dual_basis_functions=True):
    """
    Create a mesh from coordinate A to B and then to C.
    """
    n_ele = n_elements
    n_nod = order * n_ele + 1
    coordinates = np.zeros((3,n_nod))
    for i in range(3):
        coordinates[i,:] = np.linspace(A[i], B[i], n_nod)
    
    beam = []
    for i in range(n_ele):
        element = elmt.SimoBeam(
            nodes=__ele_nodes(i, order),
            n_nodes_in_mesh=n_nod,
            mesh_dof_per_node=7,
            ref_vec=reference_vector,
            coordinates=coordinates[:,__ele_nodes(i, order)],
            area=material['area'],
            elastic_modulus=material['elastic_modulus'],
            shear_modulus=material['shear_modulus'],
            inertia_primary=material['inertia_primary'],
            inertia_secondary=material['inertia_secondary'],
            inertia_torsion=material['inertia_torsion'],
            density=material['density'],
            contact_radius=material['contact_radius']
        )
        if len(possible_contact_partners) != 0:
            element.child = elmt.MortarContact(
                parent_element=element,
                n_integration_points=n_nod,
                possible_contact_partners=possible_contact_partners,
                consider_jacobian=consider_contact_jacobian,
                dual_basis_functions=dual_basis_functions

            )
        beam.append(element)
    return (coordinates, beam)
