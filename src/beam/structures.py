import numpy as np
from beam import interpolation as intp
from beam import errors


class BeamElementProperties:
    def __init__(
        self,
        length: float,
        material,
    ):
        
        # prepare empty values
        self.C = np.zeros(shape=(6,6))
        self.Arho = 0
        self.Irho = np.zeros(shape=(3,3))

        try:
            E = material['Elastic modulus']
        except KeyError:
            pass
        
        try:
            A = material['Area']
        except KeyError:
            pass
        
        try:
            G = material['Shear modulus']
        except KeyError:
            pass

        try:
            nu = material['Poisson coefficient']
        except KeyError:
            pass

        try:
            ks1 = material['Shear coefficient primary']
        except KeyError:
            pass

        try:
            ks2 = material['Shear coefficient secondary']
        except KeyError:
            pass

        try:
            It = material['Inertia torsion']
        except KeyError:
            pass

        try:
            I1 = material['Inertia primary']
        except KeyError:
            pass

        try:
            I2 = material['Inertia secondary']
        except KeyError:
            pass
        
        try:
            rho = material['Density']
        except KeyError:
            pass

        try:
            rho = material['Density']
        except KeyError:
            pass

        try:
            self.cr = material['Contact radius']
        except KeyError:
            pass
        
        # tension stiffness
        try:
            self.C[0,0] = material["EA"]
        except KeyError:
            try:
                self.C[0,0] = E * A
            except:
                raise errors.MaterialError("One of the following data is missing in material specification: 'EA', 'Area' or 'Elastic modulus'.")
            
        
        # shear stiffness - primary axis
        try:
            self.C[1,1] = material["GA1"]
        except KeyError:
            try:
                self.C[1,1] = G * A * ks1
            except:
                try:
                    self.C[1,1] = E / (2*(1+nu)) * A * ks1
                except:
                    raise errors.MaterialError("One of the following data is missing in material specification: 'GA1', 'Shear modulus', 'Poisson coefficient' or 'Shear coefficient primary'.")

        # shear stiffness - secondary axis
        try:
            self.C[2,2] = material["GA2"]
        except KeyError:
            try:
                self.C[2,2] = G * A * ks2
            except:
                try:
                    self.C[2,2] = E / (2*(1+nu)) * A * ks2
                except:
                    raise errors.MaterialError("One of the following data is missing in material specification: 'GA1', 'Shear modulus', 'Poisson coefficient' or 'Shear coefficient secondary'.")

        # torsion stiffness
        try:
            self.C[3,3] = material["GIt"]
        except KeyError:
            try:
                self.C[3,3] = G * It
            except:
                raise errors.MaterialError("One of the following data is missing in material specification: 'GIt' or 'Inertia torsion'.")

        # bending stiffness - primary axis
        try:
            self.C[4,4] = material["EI1"]
        except KeyError:
            try:
                self.C[4,4] = E * I1
            except:
                raise errors.MaterialError("One of the following data is missing in material specification: 'EI1' or 'Inertia primary'.")
            
        # bending stiffness - secondary axis
        try:
            self.C[5,5] = material["EI2"]
        except KeyError:
            try:
                self.C[5,5] = E * I2
            except:
                raise errors.MaterialError("One of the following data is missing in material specification: 'EI2' or 'Inertia secondary'.")
            
        
        # translational inertia
        try:
            self.Arho = material["Arho"]
        except KeyError:
            try:
                self.Arho = A * rho
            except:
                raise errors.MaterialError("One of the following data is missing in material specification: 'Arho' or 'Density'.")
        
        # deviatoric inertia
        try:
            self.Irho[0,0] = material["I12rho"]
        except KeyError:
            try:
                self.Irho[0,0] = rho * (I1 + I2)
            except:
                raise errors.MaterialError("One of the following data is missing in material specification: 'Density', 'Inertia primary' or 'Inertia secondary'.")
        
        # rotational inertia primary
        try:
            self.Irho[1,1] = material["I1rho"]
        except KeyError:
            try:
                self.Irho[1,1] = rho * I1
            except:
                raise errors.MaterialError("One of the following data is missing in material specification: 'Density', or 'Inertia primary'.")
        
        # rotational inertia secondary
        try:
            self.Irho[2,2] = material["I2rho"]
        except KeyError:
            try:
                self.Irho[2,2] = rho * I2
            except:
                raise errors.MaterialError("One of the following data is missing in material specification: 'Density' or 'Inertia secondary'.")


class BeamIntegrationPoint:
    """
    A class with all values, stored in integration points for a beam.

    ...

    Attributes
    ----------
    n_pts : int
        number of integration points
    loc : np.ndarray, shape=(n_pts,)
        locations of integration points on the interval [-1, 1]
    wgt : np.ndarray, shape=(n_pts,)
        integration weights
    Ndis : np.ndarray, shape=(n_nodes, n_pts)
        interpolation function matrix for displacement dof
    Nrot : np.ndarray, shape=(n_nodes, n_pts)
        interpolation function matrix for rotation dof
    rot : np.ndarray, shape=(4,n_pts)
        quaternion orientation of the cross-section
    om : np.ndarray, shape=(3,n_pts)
        curvature vector
    w : np.ndarray, shape=(3,n_pts)
        angular velocity vector
    a : np.ndarray, shape=(3,n_pts)
        angular acceleration vector
    q : np.ndarray, shape=(3,n_pts)
        external distributed line load
    f : np.ndarray, shape=(3,n_pts)
        internal distributed forces
    
    Methods
    -------
    
    """
    def __init__(
        self,
        displacement_interpolation,
        rotation_interpolation,
        points_location: np.ndarray = None,
        weights: np.ndarray = None
    ):
        self.n_pts = 0 if points_location is None else len(points_location)
        self.loc = points_location
        self.wgt = weights
        
        # pre-computed values for efficiency
        if self.n_pts > 0:
            self.N_displacement = displacement_interpolation[0](self.loc)
            self.dN_displacement = displacement_interpolation[1](self.loc)
            self.N_rotation = rotation_interpolation[0](self.loc)
            self.dN_rotation = rotation_interpolation[1](self.loc)

        self.rot = np.empty(shape=(3,4,self.n_pts))
        self.om = np.empty(shape=(3,3,self.n_pts))
        self.w = np.empty(shape=(3,3,self.n_pts))
        self.a = np.empty(shape=(3,3,self.n_pts))
        self.q = np.empty(shape=(3,6,self.n_pts))
        self.f = np.zeros(shape=(3,6,self.n_pts))


class IntegrationPoint:
    """
    A class with all values, stored in integration points for a finite element.

    ...

    Attributes
    ----------
    n_pts : int
        number of integration points
    loc : np.ndarray, shape=(n_pts,)
        locations of integration points on the interval [-1, 1]
        
    """
    def __init__(
        self,
        point_location: np.ndarray = None,
        weight: np.ndarray = None
    ):
        self.loc = point_location
        self.wgt = weight
