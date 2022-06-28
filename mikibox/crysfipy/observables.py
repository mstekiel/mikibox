from . import constants as C
from .cefmatrices import *

import numpy as np
from numpy import conj, transpose, dot, diag  
    

def boltzman_population(energies, temperature):
    '''
    Calculate the population of energy levels at given temperature based on the Boltzmann statistic.
    
    :math:`n_i = \\frac{1}{Z} e^{-E_i/k_B T}`
    
    :math:`Z = \\sum_i e^{-Ei/k_BT}`
    
    One important distinction, is that this function works with eigenvalues (energies) of eigenvectors from the whole Hilbert space, as it needs to evaluate :math:`Z` on its own. This works well for the total angular momentum Hilbert space, and does care about degeneracies.
    
    Parameters:
        energies : array_like
            List of energy levels in meV units
        tmperature : float
            Temperature at which to evaluate the statistic
            
    Returns:
        List of occupation probabilities for each energy level.
    '''
    
    p = np.exp(-energies*C.eV2K/temperature)
    Z = sum(p)
    return p / Z

def _rawneutronint(E, J2_perp, gJ, T):
    """Returns transition intensities in barn.
    
    TODO I think the Debye-Waller factor needs to be incorporated here for proper inter-Temperature comparisons

    Args:
        E (2D array of floats): matrix of energy changes corresponding to transitions in meV
        J2_perp (2D array of floats): matrix of squared J
        gJ (float): Landé factor
        T (float): temperature in **K**
    """
    r02 = C.R0 * C.R0  *1e28 # to have value in barn
    c = np.pi * r02 * gJ * gJ
    
    # Calculate the occupancy of the levels at given temperature
    prst = boltzman_population(E[0,:], T)
    
    # Multiply the matrix elements by uprobability of occupying certain level
    trans_int = J2_perp * prst[:, np.newaxis] * c  #transition intensities in barn
    
    return trans_int

def neutronint(cefion, T, Q = 'powder', Ei=1e+6):
    """
    Returns matrix of energies and inelastic neutron scattering spectral weights for all possible transitions at given temperature. The spectral weight is calculated by equation from Enderle book following Stephane Raymond article.
    
    | :math:`S(\\vec{Q},\\omega) = N (\\gamma r_0)^2 f^2_m(Q) e^{-2W(Q)} \\sum_{if} \\frac{k_f}{k_i} p_i |<\\lambda_f|J_\perp|\\lambda_i>|^2 \\delta(E_i - E_f - \\hbar \\omega)`
    | :math:`N (\\gamma r_0)^2` : ignored, acts as units.
    | :math:`f^2_m(Q)` : magnetic form factor, taken from internal tables in ``mikibox.crysfipy.Ion`` class.
    | :math:`e^{-2W(Q)}` : :math:`W(Q)` is the Debye-Waller factor. It is quite problematic, is set to 1 at the moment.
    | :math:`\\frac{k_f}{k_i}` : scaling factor calculated from energy, which is used more widely :math:`\\frac{k_f}{k_i} = \\sqrt{1-\\frac{\\Delta E}{E_i}}`. there is a minus under the square root, because positive energy transfer corresponds to neutron energy loss.
    | :math:`p_i` : Boltzmann population factor.
    | :math:`|<\\lambda_f|J_\\perp|\\lambda_i>|^2` : matrix elements, exact description depends on ``Q``, see below.

    
    The intensities are evaluated based on the :math:`|<\\lambda_f|J_\perp|\\lambda_i>|^2` matrix elements, which form a matrix :math:`|J_\perp|^2`.
    Two main cases are implemented and encoded in the :data:`Q` parameter.
    
    Parameters:
        cefion : :obj:`crysfipy.reion.CEFion` 
            Rare-earth ion in the crystal field
        T : float
            Temperature in *K*
        Q : 'powder' or ndarray, optional
            Scheme according to which :math:`|J_\perp|^2` is calculated.
            
            * powder :  :math:`|<\\lambda_f|J_\\perp|\\lambda_i>|^2 = 2/3\\sum_\\alpha |<\\lambda_f|J_\\alpha|\\lambda_i>|^2` (default).
            * (3,) float : :math:`|<\\lambda_f|J_\\perp|\\lambda_i>|^2 = \\sum_\\alpha (1-\\frac{Q_\\alpha}{Q})|<\\lambda_f|J_\\alpha|\\lambda_i>|^2`. Q is a vector representing a direction in the reciprocal space in respect to which a perpendicular projection of :math:`J` will be calculated.

            
    Returns:
        energies : ndarray
            Array containing energies of the transitions
        intensities : ndarray
            Array containing intensities of the transitions
            
    Raises:
        ValueError 
            When an invalid ``Q`` parameter is chosen, or the dimension of the ``Q`` vector is not 3. 
        RuntimeWarning
            When Q=[0,0,0], where the spectral weight is ill defined.        
    """
    
    # The way it is calculated is that the factor within the sum is calculated as a matrix, which is flattened at the end. Energy degeneracies are not taken into account, as it is easier to handle.
    
    # Magnetic form factor
    f2m = cefion.ion.mff(np.linalg.norm(Q))**2
    
    # Debye-Waller factor
    eDW = 1
    
    # Tricky way to create a 2D array of energies associated with transitions between levels
    jumps = cefion.energies - cefion.energies[np.newaxis].T
    
    
    # Calculate the |<\Gamma_f|J_\perp|\Gamma_i>|^2 matrix
    if type(Q) in [list, np.ndarray]:
        Q = np.array(Q).flatten()
        if Q.shape[0] != 3:
            raise ValueError('Dimension of the``Q`` vector is not 3')
            
        # First implementation does not seem to work well
        # Qperp_projectCEFion = ms.perp_matrix(Q)
        # J_perp = np.einsum('ij,jkl',Qperp_projectCEFion, cefion.J)
        # J2_perp = np.einsum('ijk->jk', np.square(np.abs(J_perp)))
        
        J2 = np.square(np.abs(cefion.J))
        projection = 1-(Q/np.linalg.norm(Q))**2
        
        J2_perp = np.einsum('i,ijk',projection, J2)
        
    elif Q=='powder':
        J2_perp = 2.0 / 3 * np.einsum('ijk->jk',np.square(np.abs(cefion.J)))    
        
    else:
        raise ValueError('Invalid ``Q`` parameter')
        
        
    # kf/ki factor, which is actually a matrix
    kfki = np.sqrt(1-jumps/Ei)
    
    # Occupation
    prst = boltzman_population(cefion.energies, T)
        
    # Multiply the factors, vectors and matrices properly to get spectral weight.
    Sqw = f2m * eDW * kfki * J2_perp * prst[:, np.newaxis]
    
    Denergies = jumps.flatten()
    sorting = Denergies.argsort()
    return (Denergies[sorting], Sqw.flatten()[sorting])

def magnetization(cefion, T, Hfield):
    '''
    Calculate the magnetization of the single ion in the crystal field.
    :math:`M_\\alpha = g_J \\sum_n p_n <\\lambda_n | \hat{J}_\\alpha | \\lambda_n>`
    
    '''
    
    cefion_inH = CEFion(cefion.ion, Hfield, cefion.cfp, diagonalize=True)
    
    # The diagonalized Hamiltonians' operators are already transformed into the sorted eigenvector base
    p = boltzman_population(cefion_inH.energies, T)
    M = cefion_inH.ion.gJ * np.real( np.einsum('ijj,j',cefion_inH.J,p) )  
    
    return M


def _rawsusceptibility(energy, moment, H_direction, H_size, T):
    """Returns susceptibility calculated for energy levels at given temperature"""

    prst = np.exp(-energy/T)
    Z = sum(prst);                                    # canonical partition function 
    prst = prst / Z;
    overal_moment = dot(prst, moment);
    return dot(overal_moment, H_direction.conj().transpose()) / H_size

def susceptibility(cefion, T, Hfield_direction, method):
    """
    Calculate the magnetic susceptibility at given temperature.
    
    The susceptibility is calculated as a numerical derivative of the magnetization. But it seems there are some other smart methods to calculate it, so take a look into these japanese papers.
    
    Parameters:
        ion : :obj:`crysfipy.reion.CEFion`
            Rare-earth ion in crystal field\
        T : float
            Temperature
        Hfield_direction:
            Direction of the applied magnetic field.
            
    Returns:
        List of susceptibility values calculated at given temperatures
    """

    # TODO complete rework
    # Next liens are taken from the previous version od the code
    # self.moment[:,0] = - self.ion.gJ * np.real(diag(self.Jx))
    # self.moment[:,1] = - self.ion.gJ * np.real(diag(self.Jy))                      
    # self.moment[:,2] = - self.ion.gJ * np.real(diag(self.Jz))

    if method=='magnetization':
        susceptibility = np.zeros(len(T))
        eps = 1e-8
        
        for it, temperature in enumerate(T):
            Hfield = eps * np.array(Hfield_direction)/np.linalg.norm(Hfield_direction)
            M = magnetization(cefion, temperature, Hfield)
            susceptibility[it] = np.linalg.norm(M)/eps
    else:
        raise ValueError('Unknown method to calculate magnetization.')
        
    return susceptibility
        
        
def thermodynamics(cefion, T):
    """
    Calculate the fundamental thermodynamic values as a function of temperature.
    
    This is all calculated together taking advantage of the fact that all thermodynamics can be determined from the partition function :math:`Z`, upon differentiation on :math:`\\beta`, where :math:`\\beta = \\frac{1}{k_B T}`.
    
    | Partition function: :math:`Z = \\sum_n e^{-\\beta E_n}`
    | Average energy: :math:`\\langle E \\rangle = - \\frac{\\partial Z}{\\partial \\beta}`
    | Entropy: :math:`S = k_B ( \\ln Z - \\beta \\frac{\\partial Z}{\\partial \\beta} )`
    | Heat capacity: :math:`C_V = k_B \\beta^2 \\frac{\\partial^2 Z}{\\partial \\beta^2}`
    
    Parameters:
        cefion : :obj:`crysfipy.CEFion`
            Rare-earth ion in crystal field
        T : ndarray
            Temperature

            
    Returns:
        Z, E, S CV : The partition function, average energy (internal energy), entropy and heat capacity, respectively.
    """
    Z = np.zeros(len(T))
    beta = C.eV2K/T
    
    for En in cefion.energies:
        Z += np.exp(-En*beta)
        

    dlogZ = np.gradient(np.log(Z), beta)
    d2logZ = - np.gradient(dlogZ, beta)
    
    E = -dlogZ
    S = np.log(Z)-beta*dlogZ
    Cv = beta**2 * d2logZ
    
    return Z, E, S, Cv

    