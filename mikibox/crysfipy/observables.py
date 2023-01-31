from . import constants as C
from .cefmatrices import *
from .cefion import CEFion

import numpy as np
    

def boltzman_population(energies: list[float], temperature: float) -> list[float]:
    r'''
    Calculate the population of energy levels at given temperature based on the Boltzmann statistic.
    :math:`p_i = \frac{1}{Z} e^{-E_i/k_B T}`
    :math:`Z = \sum_i e^{-Ei/k_BT}`
    One important distinction, is that this function works with eigenvalues (energies) from the whole Hilbert space,
    as it needs to evaluate :math:`Z` on its own. This works well for the total angular momentum Hilbert space, and does care about degeneracies.
    
    Args:
        energies: List of energy levels in meV units
        temperature: Temperature at which to evaluate the statistic

    Returns:
        List of occupation probabilities for each energy level.
    '''
    
    p = np.exp(-np.array(energies - min(energies))*C.meV2K/temperature)
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

def neutronint(cefion: CEFion, temperature: float, Q: tuple , scheme: str, Ei: float=1e+6):
    r"""
    Returns matrix of energies and inelastic neutron scattering spectral weights for all possible transitions at given temperature.
    
    The spectral weight is calculated by equation from Enderle book following Stephane Raymond article.
    
    | :math:`S(\vec{Q},\omega) = N (\gamma r_0)^2 f^2_m(Q) e^{-2W(Q)} \sum_{if} \frac{k_f}{k_i} p_i |<\lambda_f|J_\perp|\lambda_i>|^2 \delta(E_i - E_f - \hbar \omega)`
    
    where:

    | :math:`N (\gamma r_0)^2` : ignored, acts as units.
    | :math:`f^2_m(Q)` : magnetic form factor, taken from internal tables in ``mikibox.crysfipy.Ion`` class.
    | :math:`e^{-2W(Q)}` : :math:`W(Q)` is the Debye-Waller factor. It is quite problematic, is set to 1 at the moment.
    | :math:`\frac{k_f}{k_i}` : scaling factor calculated from energy, which is used more widely :math:`\frac{k_f}{k_i} = \sqrt{1-\frac{\Delta E}{E_i}}`. there is a minus under the square root, because positive energy transfer corresponds to neutron energy loss.
    | :math:`p_i` : Boltzmann population factor.
    | :math:`|<\lambda_f|J_\perp|\lambda_i>|^2` : matrix elements, exact description depends on ``Q``, see below.

    
    The intensities are evaluated based on the :math:`|<\lambda_f|J_\perp|\lambda_i>|^2` matrix elements, which form a matrix :math:`|J_\perp|^2`.
    Two main cases are implemented and encoded in the :data:`Q` parameter.
    
    Args:
        cefion: :class:`CEFion` Rare-earth ion in the crystal field
        tempreature:  Temperature in *K*
        Q :  List of Q vectors used to evaluate the spectral weight
        
        scheme : 'powder', 'single-crystal':
            Scheme according to which :math:`|J_\perp|^2` is calculated.
            
            * powder :  :math:`|<\lambda_f|J_\perp|\lambda_i>|^2 = 2/3\sum_\alpha |<\lambda_f|J_\alpha|\lambda_i>|^2` (default).
            * (single-crystal : :math:`|<\lambda_f|J_\perp|\lambda_i>|^2 = \sum_\alpha (1-\frac{Q_\alpha}{Q})|<\lambda_f|J_\alpha|\lambda_i>|^2`. Q is a vector representing a direction in the reciprocal space in respect to which a perpendicular projection of :math:`J` will be calculated.

            
    Returns:
        energies: Array containing energies of the transitions
        intensities: Array containing intensities of the transitions
            
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
    if scheme=='single-crystal':
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
    elif scheme=='powder':
        J2_perp = 2.0 / 3 * np.einsum('ijk->jk',np.square(np.abs(cefion.J)))    
    else:
        raise ValueError('Invalid ``Q`` parameter')
        
        
    # kf/ki factor, which is actually a matrix
    kfki = np.sqrt(1-jumps/Ei)
    
    # Occupation
    transition_probs = boltzman_population(cefion.energies, temperature)
        
    # Multiply the factors, vectors and matrices properly to get spectral weight.
    Sqw = f2m * eDW * kfki * J2_perp * transition_probs[:, np.newaxis]
    
    Denergies = jumps.flatten()
    sorting = Denergies.argsort()
    return (Denergies[sorting], Sqw.flatten()[sorting])


def magnetization(cefion: CEFion, temperature: float, Hfield: tuple[float,float,float]):
    r'''
    Calculate the magnetization of the single ion in the crystal field.
    Returned value is in :math:`\mu_B` units.

    :math:`M_\alpha = g_J \sum_n p_n |<\lambda_n | \hat{J}_\alpha | \lambda_n>|`

    Args:
        cefion: Rare-earth ion in the crystal field
        temperature: temperature at which to calculate magnetization   
        Hfield: Applied magnetic field. Both direction and vlue are important.
    '''
    
    # Solve the Hamiltonian of a copy of the given ion
    cefion_inH = CEFion(cefion.ion, Hfield, cefion.cfp, diagonalize=True)

    # The diagonalized Hamiltonians' operators are already transformed into the sorted eigenvector base
    p = boltzman_population(cefion_inH.energies, temperature)
    M = cefion_inH.ion.gJ * np.abs( np.einsum('ijj,j', cefion_inH.J, p) )
    
    return M


def susceptibility(cefion: CEFion, temperatures: list[float], Hfield_direction: tuple[float,float,float], method: str='perturbation') -> list[float]:
    r"""
    Calculate the magnetic susceptibility at listed temperatures, based on one of the implemented methods.
    
    
    `perturbation`:
    Based on assuming an infinitezimal applied field that perturbes the Hamiltonian, the eigenstates
    are calculated by means od perturbation theory and given by formula:

    | :math:`\chi_{CEF} = (g_J \mu_B)^2 \left[ \sum_{n,m \neq n}  p_n \frac{1-exp(-\Delta_{m,n}/k_B T)}{\Delta_{m,n}} |<\lambda_m|J|\lambda_n>|^2  +  \frac{1}{k_B T} \sum_{n} p_n |<\lambda_n|J|\lambda_n>|^2 \right]`
    
    where
    
    | :math:`g_J \mu_B` : Lande factor, Bohr magneton.
    | :math:`p_n` : Ptobabitlity of occupying the energy level :math:`\lambda_n` at given temperature.
    | :math:`\Delta_{m,n}` : Energy of transition between the :math:`\lambda_n` and :math:`\lambda_m` levels.
    | :math:`<\lambda_m|J|\lambda_n>` : J matrix elements.




    `magnetization`:
    Based on calculating a numerical derivative of the magnetization, with a very small magnetic field

    | :math:`\chi_{CEF} = \left. \frac{\partial \vec{M}}{\partial \vec{H}} \right|_{\vec{H}=\epsilon}`

    Magnetization is calculated internally by :func:`magnetization`. :math:`\epsilon = 10^{-8}`

    ALL BELOW MADE A CRASH FOR ANOTER SYSTEM
    In calculations a dangerous trick is used. The :math:`\frac{1-exp(-\Delta_{m,n}/k_B T)}{\Delta_{m,n}}` 
    factor is computed as a matrix, so naturally it will be divergent on the diagonal, since :math:`\Delta_{m,n}=0` there. 
    This raises a RuntimeWarning which is ignored , and the diagonal is replaced with the :math:`1/k_B T` values, and the two summations are bundled together. 
    This feels dirty, but works so far, even for spin-half systems, which have degenerated energy levels.
    Maybe because the transitions between degenerated levels are not exactly 0 in the calculations (1e-15 rather).

    
    Parameters:
        cefion : :class:`crysfipy.CEFion`
            Rare-earth ion in crystal field\
        temperatures : ndarray
            Array ocntaining temperatures at which to compute the susceptibility.
        Hfield_direction:
            Direction of the applied magnetic field. Value can be arbitrary, it is normalized in the code.
        method: optional, 'perturbation', 'magnetization'
            Method by which to calculate susceptibility. Old implementation 
            
    Returns:
        List of susceptibility values calculated at given temperatures. In the units of :math:`\mu_B^2`.
    """
    


    if method=='magnetization':
        susceptibility = np.zeros(len(temperatures))
        eps = 1e-8  
        
        for it, temperature in enumerate(temperatures):
            Hfield = eps * np.array(Hfield_direction)/np.linalg.norm(Hfield_direction)
            M = magnetization(cefion, temperature, Hfield)
            susceptibility[it] = np.linalg.norm(M)/eps
    elif method=='perturbation':
        susceptibility = np.empty(len(temperatures))
        #susceptibility.fill(np.nan)

        for it, temperature in enumerate(temperatures):
            # Tricky way to create a 2D array of energies associated with transitions between levels
            jumps = cefion.energies - cefion.energies[np.newaxis].T

            # Clean up
            jumps_with_zero_energy = np.where(np.abs(jumps)< C.numerical_zero)
            jumps_with_positive_energy = np.where(jumps>0)
            # jumps_with_negative_energy = np.where(jumps<0) # obsolete but left for clarifty, these will be set to zero

            # Define the transition matrix taking into account the diagonal and negative energy jumps
            Tmx = np.zeros(np.shape(jumps))
            Tmx[jumps_with_zero_energy] = 1*C.meV2K/temperature
            Tmx[jumps_with_positive_energy] = -np.expm1(-jumps[jumps_with_positive_energy]*C.meV2K/temperature)/jumps[jumps_with_positive_energy]

            # Include the Boltzmann factor
            transition_probs = boltzman_population(cefion.energies, temperature)
            Tmx = Tmx * transition_probs[:, np.newaxis]

            # Calculate the J^2 matrix
            J2 = np.square(np.abs(cefion.J))
            J2_directed = np.einsum('i,ijk',Hfield_direction, J2)/np.linalg.norm(Hfield_direction)
            
            susceptibility[it] = (cefion.ion.gJ)**2 * np.sum(Tmx * J2_directed)
    else:
        raise ValueError('Unknown method to calculate magnetization.')
        
    return susceptibility
        
        
def thermodynamics(cefion, T):
    r"""
    Calculate the fundamental thermodynamic values as a function of temperature.
    
    These functions are calculated alltogether taking advantage of the fact that thermodynamics can be determined from the partition function :math:`Z`, upon differentiation on :math:`\beta`, where :math:`\beta = \frac{1}{k_B T}`.
    
    | Partition function: :math:`Z = \sum_n e^{-\beta E_n}`
    | Average energy: :math:`\langle E \rangle = - \frac{\partial Z}{\partial \beta}`
    | Entropy: :math:`S = k_B ( \ln Z - \beta \frac{\partial Z}{\partial \beta} )`
    | Heat capacity: :math:`C_V = k_B \beta^2 \frac{\partial^2 Z}{\partial \beta^2}`
    
    Parameters:
        cefion : :obj:`crysfipy.CEFion`
            Rare-earth ion in crystal field
        T : ndarray
            Temperature

            
    Returns:
        Z, E, S CV : The partition function, average energy (internal energy), entropy and heat capacity, respectively.
    """
    Z = np.zeros(len(T))
    beta = C.meV2K/T
    
    for En in cefion.energies:
        Z += np.exp(-En*beta)
        

    dlogZ = np.gradient(np.log(Z), beta)
    d2logZ = - np.gradient(dlogZ, beta)
    
    E = -dlogZ
    S = np.log(Z)-beta*dlogZ
    Cv = beta**2 * d2logZ
    
    return Z, E, S, Cv

    