import numpy as np
#import mikibox as ms

def lattice_pars_from_UB(UB):
    '''
    Determine the lattice parameters from the UB matrix.
    
    The assumed convention of the UB matrix are:
        1. U is an orthonormal matrix
        2. B is an upper triangle matrix which transform (hkl)->[kx,ky,kx].
        3. B is in 1/Angstroem units without 2pi and wavelength factors.
    '''
    U, B = np.linalg.qr(UB)

    a1,a2,a3 = np.linalg.inv(B)

    a = np.sqrt(np.dot(a1,a1))
    b = np.sqrt(np.dot(a2,a2))
    c = np.sqrt(np.dot(a3,a3))

    alp = np.degrees(np.arccos(np.dot(a2,a3)/(b*c)))
    bet = np.degrees(np.arccos(np.dot(a1,a3)/(a*c)))
    gam = np.degrees(np.arccos(np.dot(a1,a2)/(a*b)))

    return (a,b,c, alp,bet,gam)