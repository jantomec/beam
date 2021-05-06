# -*- coding: utf-8 -*-
""" Module for simple mesh generation.

This modules includes functions that help with commonly shape meshes. It
is not necessary to use these functions for the beam application to work
however, it makes the code much easier to read. This is why it is
preferred to always try and implement a general form of the mesh here
first.

Functions:
    line_mesh(A, B, n_elements, order, material, reference_vector, possible_contact_partners=[], consider_contact_jacobian=False, dual_basis_functions=True)
    n_point_mesh(points, n_elements, order, material, reference_vector, possible_contact_partners=[], consider_contact_jacobian=False, dual_basis_functions=True):

Examples:
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
"""

import numpy as np
import elements as elmt


def __ele_nodes(ele_id, n_nodes_per_ele):
    return np.array([
        n_nodes_per_ele*ele_id+j for j in range(n_nodes_per_ele+1)
    ], dtype=int)

def line_mesh(A, B, n_elements, order, material, reference_vector, starting_node_index=0, possible_contact_partners=[], consider_contact_jacobian=False, dual_basis_functions=True):
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
            nodes=starting_node_index+__ele_nodes(i, order),
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
                dual_basis_functions=dual_basis_functions
            )
            element.child.consider_jacobian = consider_contact_jacobian
        beam.append(element)
    return (coordinates, beam)

def n_point_mesh(points, n_elements, order, material, reference_vector, starting_node_index=0, possible_contact_partners=[], consider_contact_jacobian=False, dual_basis_functions=True):
    """
    Create a mesh from a list of points by connecting them in a sequence
    (P1 -- P2 -- P3 -- ... -- PN).

    # Parameters:
    points ...................... points in 3D
    n_elements .................. a list containing the number of elements for each segment
    order ....................... element order (polynomial interpolation order)
    material .................... dictionary with material properties
    reference_vector ............ a vector to define the orientation of the cross-section
    possible_contact_partners ... a list of elements containing elements, that might get in contact with the elements from this mesh
    consider_contact_jacobian ... a boolean saying if the Jacobian should be considered in the contact definition
    dual_basis_functions ........ a boolean saying if the Lagrange multiplier field should be interpolated with dual shape functions or with Lagrange polynomials
    """
    
    assert points.shape[1] == len(n_elements) + 1, 'Number of points should be one greater then the length of n_elements list.'
    n_ele = np.array(n_elements)
    n_nod = order * np.sum(n_ele) + 1
    coordinates = np.zeros((3,n_nod))
    for i in range(len(n_ele)):
        n1 = order*np.sum(n_ele[:i])
        n2 = order*np.sum(n_ele[:i])+order*n_ele[i]
        for j in range(n1, n2):
            for k in range(3):
                coordinates[k,j] = points[k,i] + (points[k,i+1] - points[k,i]) * (j - n1) / (n2 - n1)
    coordinates[:,-1] = points[:,-1]

    beam = []
    for i in range(np.sum(n_ele)):
        element = elmt.SimoBeam(
            nodes=starting_node_index+__ele_nodes(i, order),
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
                dual_basis_functions=dual_basis_functions
            )
            element.child.consider_jacobian = consider_contact_jacobian
        beam.append(element)
    return (coordinates, beam)
