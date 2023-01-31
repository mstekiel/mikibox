import numpy as np
import mikibox as ms

from typing import Union


import warnings
warnings.filterwarnings("error")

class Lattice:
    '''
    Object representing the lattice of a crystal.
    
    The crystallographic conventions should be cleared out.
    
    The main idea is that the orientation can be changed, but the lattice type and parameters no.
    
    Attirbutes:
        lattice_parameters: ndarray((6))
            List containing `a,b,c,alpha, beta, gamma` lattice parameters. Lengths in angstroems, angles in degrees.
        A : ndarray((3,3))
            Transforms a real lattice point into an orthonormal coordinates system. Upper triangle matrix.
            [u,v,w] -> [x,y,z] (Angstroems)
        B : ndarray((3,3))
            Transforms a reciprocal lattice point into an orthonormal coordinates system. Upper triangle matrix.
            (h,k,l) -> [kx,ky,kz] (1/Angstroem)
        U : ndarray((3,3))
            Orientation matrix that relates the orthonormal, reciprocal lattice coordinate system into the diffractometer/lab coordinates
            [kx,ky,kz]_{crystal} -> [kx,ky,kz]_{lab}
        UB : ndarray((3,3))
            Transforms a reciprocal lattice point into lab coordinate system.
            (h,k,l) -> [kx,ky,kz]_{lab}
    '''

    def __init__(self, lattice_parameters: list[float], orientation: tuple=None):
        '''
        Lattice parameters are: a,b,c,alpha,beta,gamma.
        
        Orientation: None, (hkl1, hkl2)
            Sample orientation. 'None' initializes with identity matrix, pair of hkl's puts the first one along `x` and the second one in the `xy` plane.
        '''
        self.lattice_parameters = lattice_parameters
        #self.G = self.metricTensor(a,b,c,alpha,beta,gamma)
        
        # Transforms real lattice points into orthonormal coordinate system.
        self.A = self.constructA(lattice_parameters)
        
        # Transforms reciprocal lattice points into orthonormal coordinate system.
        self.B = self.constructB(lattice_parameters)
        
        # Initialize the orientation, U and the UB matrix within the wrapper.
        self.updateOrientation(orientation)

                
        
    def __str__(self):
        return str(self.lattice_parameters)
    
    def constructA(self, lattice_parameters: list[float]) -> np.ndarray:
        # Construct the A lattice as crystal axes in orthonormal system, ie a||x, b in xy plane, c accordingly.
        a,b,c,alpha,beta,gamma = lattice_parameters
        
        bx = b*np.cos(np.radians(gamma))
        by = b*np.sin(np.radians(gamma))
        
        cx = c*np.cos(np.radians(beta))
        cy = c*(np.cos(np.radians(alpha))-np.cos(np.radians(gamma))*np.cos(np.radians(beta)))/np.sin(np.radians(gamma))
        cz  = np.sqrt(c*c-cx*cx-cy*cy)
        
        return np.array([[a,bx,cx],[0,by,cy],[0,0,cz]])
    
    def constructB(self, lattice_parameters: list[float]) -> np.ndarray:
        # Construction based on the perpendicularity.
        A = self.constructA(lattice_parameters)
        B = np.linalg.inv(A)

        # Three things
        # 1. Transpose B to get column-vector representation
        # 2. I want to keep convention where a* is along cartesian x, b* in xy plane and c* has positive z-component.
        #    This is problematic for hexagonal, monoclinic and triclinic systems, since I loose the perpendicularity between real and reciprocal axes.
        #    Lets see if it will raise some future problems.
        
        _, B = np.linalg.qr(B.T)
        
        # Align a* along x
        if B[0,0]<0:
            B = np.dot(ms.Rz(np.pi), B)
            
        # Make c* have positive z-component
        if B[2,2]<0:
            B = np.dot(ms.Rx(np.pi), B)
            
        return B
        
    def constructU(self, orientation: Union[None, tuple, np.ndarray]) -> np.ndarray:
        '''
        Construct the orientation matrix U. Different schemes are allowed depending on the type of the `orientation` argument.
        
        orientation : None
            Identity matrix
        orientation : hkl_tuple
            The chosen hkl is put perpendicular to the scattering plane, i.e. along the `z` axis.
        orientation : (hkl1_tuple, hkl2_tuple)
            hkl1 is put along the `x` axis and hkl2 in the `xy` plane.
        orientation : ndarray(3,3)
            U is given directly as an argument.
        '''
                
        # If the orientation is not None the U matrix heas to be updated
        if orientation == None:
            # Initial orientation is as given by B
            U = np.eye(3,3)
        elif np.shape(orientation) == (3,):
            # Single vector
            hkl = orientation
            n = np.dot(self.B, hkl)
            U = ms.rotate( np.cross(n, [0,0,1]), ms.angle(n, [0,0,1]) )
        elif np.array(orientation).shape == (2,3):
            # Two vectors
            hkl1, hkl2 = orientation
            n1 = np.dot(self.B, hkl1)
            n2 = np.dot(self.B, hkl2)
            
            # This rotation puts hkl1 along `x`
            R1 = ms.rotate( np.cross(n1, [1,0,0]), ms.angle([1,0,0], n1) )

            # Find the angle necessary to put hkl2 in `xy` plane
            n3 = np.dot(R1, n2)
            beta2 = ms.angle(n3, [n3[0],n3[1],0]) * np.sign(-n3[2])
            R2 = ms.rotate( [1,0,0], beta2 )
            
            U = np.dot(R2, R1)
        elif np.array(orientation).shape == (3,3):
            U = np.array(orientation)
        else:
            raise ValueError('Wrong orientation argument for initializing the Lattice object.')
            

        return U
        
    def updateOrientation(self, orientation: Union[None, tuple, np.ndarray]):
        '''
        Update the orientation matrix of the Lattice, together with the underlying UB matrix.
        
        Raises Warning if the new matrix is not orthonormal
        '''
        
        newU = self.constructU(orientation)
        
        assert np.shape(newU) == (3,3)
        
        try:
            np.testing.assert_almost_equal(np.dot(newU[0],newU[0]), 1)
            np.testing.assert_almost_equal(np.dot(newU[1],newU[1]), 1)
            np.testing.assert_almost_equal(np.dot(newU[2],newU[2]), 1)
        except AssertionError:
            raise Warning('The new orientation matrix does not seem to be row-normalized')
            
        try:
            np.testing.assert_almost_equal(np.dot(newU[0],newU[1]), 0)
            np.testing.assert_almost_equal(np.dot(newU[0],newU[2]), 0)
            np.testing.assert_almost_equal(np.dot(newU[1],newU[2]), 0)
        except AssertionError:
            raise Warning('The new orientation matrix does not seem to be orthogonal')
            
        self.U = newU
        self.UB = np.dot(newU, self.B)
        self._current_orientation = orientation
        
        return
    
    def hkl2Q(self, hkl: Union[tuple, list]) -> Union[tuple, list]:
        '''
        Calculate the coordinates in the reciprocal space in the [kx,ky,kz] basis in :math:`1/\\A` units.
        
        Parameters:
            hkl : array_like
                Miller indices or list of Miller indices
                
        Returns: ndarray
            Vector in reciprocal space or list of vectors in reciprocal space.
        '''
        
        hkl = np.array(hkl)
        
        # hkl is a single vector
        if hkl.shape == (3,):
            out = np.dot(self.UB, hkl)
        elif hkl.shape[1] == 3:
            out = np.einsum('kj,ij', self.UB, hkl)
        else:
            raise IndexError('Incompatible dimension of the hkl array. Should be (3,) or (N,3).')
        
        return out
        
    def Q2hkl(self, Q: Union[tuple, list]) -> Union[tuple, list]:
        '''
        Calculate the Miller indices based on the reicprocal space coordinates.
        
        Parameters:
            Q : array_like
                Resiprocal space coordinates or list of thereof.
                
        Returns: ndarray
            Miller indices or list of Miller indices.
        '''
        
        Q = np.array(Q)
        
        # Q is a single vector
        if Q.shape == (3,):
            out = np.dot(np.linalg.inv(self.UB), Q)
        elif Q.shape[1] == 3:
            out = np.einsum('kj,ij', np.linalg.inv(self.UB), Q)
        else:
            raise IndexError('Incompatible dimension of the Q array. Should be (3,) or (N,3).')
        
        return out
        
    def scattering_angle(self, hkl: Union[tuple, list], wavelength: float) -> Union[tuple, list]:
        '''
        Calculate the scattering angle otherwise known as two-theta from the Miller indices.
        
        Parameters:
            hkl : array_like
                Miller indices or list of Miller indices
            wavelength : float
                Wavelength of the incoming wave in Angstroems.
                
        Returns: ndarray
            Scattering angle or a list of scattering angles.
        '''
        
        hkl = np.array(hkl)

        # hkl is a single vector
        if hkl.shape == (3,):
            Q = np.dot(self.B, hkl)
            Q_lengths = np.linalg.norm(Q)
        elif hkl.shape[1] == 3:
            Q = np.einsum('kj,ij', self.B, hkl)
            Q_lengths = np.linalg.norm(Q, axis=1)
        else:
            raise IndexError('Incompatible dimension of the Q array. Should be (3,) or (N,3).')
        
        y = wavelength*Q_lengths/2
        try:
            theta = np.arcsin(y)
        except RuntimeWarning:
            raise ValueError('Wavelength too long to be able to reach the selected hkl.')
            
        return 2*theta
        
    def is_in_scattering_plane(self, hkl: tuple) -> bool:
        '''
        Test whether the given hkl is in the scattering plane i.e. `xy` plane.
        '''
        # XY is the scattering plane, to be in the scattering plane the z component must be small.
        v = self.hkl2Q(hkl)
        v = v/ms.norm(v)
        
        return v[2]<1e-7