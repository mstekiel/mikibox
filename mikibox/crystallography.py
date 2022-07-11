import numpy as np

class Lattice:

    def __init__(self, a,b,c,alpha,beta,gamma):
        self.G = self.metricTensor(a,b,c,alpha,beta,gamma)
        self.U = self.U(a,b,c,alpha,beta,gamma)

    def metricTensor(self,a,b,c,alpha,beta,gamma):
        return np.array([[a**2,0,0],[0,b**2,0],[0,0,c**2]])
    
    def U(self,a,b,c,alpha,beta,gamma):
        return np.array([[1/a,0,0],[0,1/b,0],[0,0,1/c]])
    
    def hkl2k(self, hkl):
        '''
        Calculate the coordinates in the reciprocal space in the [kx,ky,kz] basis in :math:`1/\\A` units.
        
        Parameters:
            hkl : array_like
                Bragg indices or list of Bragg indices
                
        Returns: ndarray
            Vector in reciprocal space or list of vectors in reciprocal space.
        '''
        
        hkl = np.array(hkl)
        
        # hkl is a single vector
        if hkl.shape == (3,):
            out = np.dot(self.U, hkl)
        elif hkl.shape[1] == 3:
            out = np.einsum('kj,ij', self.U, hkl)
        else:
            raise IndexError('Incompatible dimension of the hkl array. Should be (3,) or (N,3).')
        
        return out
    
    
# Additional functions

def lattice_pars_from_UB(UB):
    U, B = np.linalg.qr(UB)

    a1,a2,a3 = np.linalg.inv(B)

    a = np.sqrt(np.dot(a1,a1))
    b = np.sqrt(np.dot(a2,a2))
    c = np.sqrt(np.dot(a3,a3))

    alp = np.degrees(np.arccos(np.dot(a2,a3)/(b*c)))
    bet = np.degrees(np.arccos(np.dot(a1,a3)/(a*c)))
    gam = np.degrees(np.arccos(np.dot(a1,a2)/(a*b)))

    return ((a,b,c), (alp,bet,gam))