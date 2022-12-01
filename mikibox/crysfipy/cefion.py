# from .. import functions as ms

from . import constants as C
from . import CEFpars, Ion
from .cefmatrices import *

import numpy as np
# from numpy import conj, transpose, dot, diag
# import numbers


class CEFion:
    r'''
    Object representing a rare-earth ion in CF potential. It is internally calculated in the meV units.
    
    Parameters:
        ion : string
            Name of the ion. They are tabulated in :obj:`ion` with their parameters.
        Hfield : 1D array of floats
            External magnetic field in *T* units.
        cfp : ``crysfipy.CEFpars``
            Crystal field parameters
        diagonalize : bool, optional
            If true (default) then it automatically diagonalizes Hamiltonian, calculates energy levels and sorts all matrices so that the first eigenvector corresponds to the lowest energy level and the last one to the highest.

    Examples:
        
        TODO
        ce = CEFion("Ce", [0,0,0], ["T", 10])
        print(ce)
        Energy levels:
        E(0) =	0.0000	 2fold-degenerated
        E(1) =	3600.0000	 4fold-degenerated
        
    Attributes:
        ion : ``crysfipy:ion``
            The ``ion`` object that represents an isolated rare-earth ion.
        Jval : int/2
            The J value corresponding to the L+S quantum numbers. Not to be confused with the ``J`` operator.
        Hfield : array_like
            Vector in real space corresponding to the external magnetic field in T units.
        cfp : ``crysfipy.reion.cfp``
            Crystal field parameters
        hamiltonian : ndarray
            Hamiltonian operator. :math:`\hat{\mathcal{H}} = \sum_{ij} B_i^j \hat{O}_i^j + g_J (H_x \hat{J}_x + H_y \hat{J}_y + H_z \hat{J}_z)`
        Jx, Jy, Jz : ndarray
            Matrices representing the prinicpal quantum operators :math:`\hat{J}_\\alpha`` with :math:`\\alpha=[x,y,z]`.
        J : ndarray
            Total angular momentum vector operator :math:`\hat{J}=[\hat{J}_x, \hat{J}_y, \hat{J}_z]`. It's multidimensionality takes some time to get used to.
        energies : ndarray
            Eigenenergies of the Hamiltonian.
        eigenvectors : ndarray
            Eigenvectors of the Hamiltonian.
        degeneracies : list
            List containing entries like [energy, degeneracy], which describes how degenerated is the given energy level.
        freeionkets : list
            List of kets corresponding to the J basis of the free ion problem.
    '''

    def __init__(self, ion: Ion, Hfield: tuple, cfp: CEFpars, diagonalize: bool=True):
        self.ion = ion
        self.Jval = self.ion.J
        Jval = self.Jval
        
        # TODO check if the Hfield parameter is appropriate
        self.Hfield = np.array(Hfield)
            
        # Assign CEF parameters as CEFpars class
        self.cfp = cfp
        

        # Prepare the rest of the fields based on the main input parameters

        # Main Ji matrices
        self.Jx = J_x(Jval); 
        self.Jy = J_y(Jval);
        self.Jz = J_z(Jval);
        self.J = [self.Jx, self.Jy, self.Jz]
        
        # matrix with projection of moments to x, y, z directions for all J2p1 levels
        self.moment = np.zeros((int(2*Jval+1),3), float)  
        
        # Prepare a list os kets that form the basis
        if Jval%1==0:
            self.freeionkets = [f'|{int(x)}>' for x in np.linspace(Jval,-Jval,int(2*Jval+1))]
        elif Jval%1==0.5:
            self.freeionkets = [f'|{int(2*x):d}/2>' for x in np.linspace(Jval,-Jval,int(2*Jval+1))]
        
        
        # THE HAMILTONIAN
        H = -C.uB * self.ion.gJ * np.einsum('ijk,i', self.J, self.Hfield)
        
        # Store Stevens operators in the dictionary containing pointers tu functions
        StevensOperator = { "B20":O_20, "B22":O_22,"B2m2":O_2m2,\
                            "B40":O_40, "B42":O_42,"B4m2":O_4m2, "B43":O_43,"B4m3":O_4m3, "B44":O_44,"B4m4":O_4m4, \
                            "B60":O_60, "B62":O_62,"B6m2":O_6m2, "B63":O_63,"B6m3":O_6m3, "B64":O_64,"B6m4":O_6m4, "B66":O_64,"B6m6":O_6m6}
        for Bij_name, Bij_value in zip(self.cfp.B_names, self.cfp.B_values):
            H += Bij_value * StevensOperator[Bij_name](Jval)
            

            
        # Lets take cc of everything to get rid of complex eigenvalues.
        # Another idea is to take just cc of the Zeeman part which sometimes makes problems
        self.hamiltonian = (H + H.conj())/2
        
        # Diagonalize the Hamiltonian
        if (diagonalize):
            self.diagonalize()
    
    def diagonalize(self, sortWithE=True, shiftToZero=True):
        """
        Diagonalize the Hamiltonian, and change to the sorted eigenvector base. The default sorting is done according to eigenenergies, so that the first vector [1,0,...,0] is the lowest eigenstate, and the last one [0,...,0,1] is the highest one. Changing the base greatly faiclitates further calculations based on the evaluation of matrix elements.
        
        It updates the attributes: {``Jx``,``Jy``,``Jz``,``J``,``energies``,``eigenvectors``,``degeneracies``}.
        
        Returns:
            None
            
        Raises:
            ValueError 
                If the calculated eigenenergies are not real.
                
        """
    
        # Diagonalize the Hamiltonian
        E, U = np.linalg.eig(self.hamiltonian);
        
        if sum(np.iscomplex(E)) > 0:
            raise ValueError('Final energies are complex!')
        
        self.energies = np.real(E) - int(shiftToZero)*min(np.real(E))     # shift to zero level

        # TODO check if U is orthogonal based on comparison with QR decomposition
        self.eigenvectors = U
        

        # Sorting
        if sortWithE:
            sortedIndices = self.energies.argsort()
        else:
            sortedIndices = np.range(Jval)
            

        self.eigenvectors = self.eigenvectors[:,sortedIndices]
        self.energies =  self.energies[sortedIndices]
        
        
        
        # Change the basis of principal operators to the eigenstate basis with specified sorting scheme
        self.Jx = dot(dot(self.eigenvectors.conj().transpose(), self.Jx), self.eigenvectors)
        self.Jy = dot(dot(self.eigenvectors.conj().transpose(), self.Jy), self.eigenvectors)
        self.Jz = dot(dot(self.eigenvectors.conj().transpose(), self.Jz), self.eigenvectors)
       
        self.J = np.array([self.Jx, self.Jy, self.Jz])

        

        #calculate degeneracy
        deg_e = []
        levels = 0
        deg_e.append([self.energies[0], 0])
        for x in self.energies:
            if not np.isclose(deg_e[levels][0], x):
                levels+=1
                deg_e.append([x, 1])
            else:
                deg_e[levels][1] += 1

        self.degeneracies = np.array(deg_e)   
        
    def __str__(self, precision=4):
        """
        Nice printout of calculated parameters
        
        Parameters:
            precision : int, optional
                How many significant digits should be shown for printout of energy and eigenvector coefficients.
        """
        ret = ""
        ret += "Energy levels and corresponding eigenfunctions:\n"
               
        n = 0
        levels = []
        for level,x in enumerate(self.degeneracies):
            level_str = ''
            energy = x[0]
            degeneracy = int(x[1])
            level_str += f'E({level:d}) =\t{energy:.{precision}} meV\t{degeneracy:2d}fold-degenerated\n'
            
            # List degenerated eigenvectors
            for ev in self.eigenvectors.T[n:n+degeneracy]:
                ev_components = []
                for c,ket in zip(ev,self.freeionkets):
                    if np.abs(c) > 1/np.power(10,precision):    # Arbitrary tolerance
                        ev_components.append(f'({c:.{precision}f}){ket}')
                        
                level_str += f'ev_{level}: ' + ' + '.join(ev_components) + '\n'

            levels.append(level_str)
            n += degeneracy
        
        ret += '\n'.join(levels)
        return ret