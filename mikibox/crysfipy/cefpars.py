import numbers
import numpy as np
from . import constants as C

class CEFpars:
    r"""
    Class representing set of crystal field parameters.

    It simplifies the creation of the CF parameter sets considering the point group of the ion.
    The symmetry restrictions follow from https://www2.cpfs.mpg.de/~rotter/homepage_mcphase/manual/node133.html
    It also allows to look up the symmetry restrictions and nice printing of parameters.
    For the lookup table between notations on point groups see: https://en.wikipedia.org/wiki/Point_group

    Special care was taken for the cubic space group, as its symmetry imposes some algebraic
    relations between the non-zero elements.    

    Attributes:
        pointGroup : string
            Name of the point group symmetry
        lattice
            Type of the Bravais lattice
        B_names : array(string)
            List of names of the non-zero Stevens operators
        B_values : array(float)
            List of values of the CEF parameters, corresponding to the operators listed in `B_names`
           
    Examples:
        Create set of CF parameters by named parameters:

        >>> print(CEFpars('C4h', [-0.696, 4.06e-3, 0.0418, 0, 4.64e-6, 8.12e-4, 1.137e-4], 'K'))
        Set of CEF parameters for C4h point group in a tetragonal lattice
        B20 = -0.05998 meV
        B40 = 0.0003499 meV
        B44 = 0.003602 meV
        B4m4 = 0.0 meV
        B60 = 3.998e-07 meV
        B64 = 6.997e-05 meV
        B6m4 = 9.798e-06 meV
    """
    
    def allowed_Bpars(self, pointGroup):
        # triclinic systems not implemented
        
        # List of allowed parameters is compiled based on McPhase manual
        # https://www2.cpfs.mpg.de/~rotter/homepage_mcphase/manual/node133.html
        
        r2 = [  "B20", "B22","B2m2",\
                "B40", "B42","B4m2", "B43","B4m3", "B44","B4m4", \
                "B60", "B62","B6m2", "B63","B6m3", "B64","B6m4", "B66","B6m6"]
        r3 = ["B20", "B22", "B40", "B42",  "B44", "B60", "B62", "B64", "B66"]
        r4 = ["B20", "B40", "B44", "B4m4", "B60", "B64", "B6m4"]
        r5 = ["B20", "B40", "B44", "B60",  "B64"]
        r6 = ["B20", "B40", "B43", "B4m3", "B60", "B63","B6m3", "B66","B6m6"]
        r7 = ["B20", "B40", "B43", "B60",  "B63", "B66"]
        r8 = ["B20", "B40", "B60", "B66",  "B6m6"]
        r9 = ["B20", "B40", "B60", "B66"]
        r10= ["B40", "B44", "B60", "B62", "B64", "B66"]
        r11= ["B40", "B44", "B60", "B64"]
    
        allowed_Bpars = {
            'C2':r2, 'Cs':r2, 'C2h':r2,\
            'C2v':r3,'D2':r3, 'D2h':r3,\
            'C4':r4, 'S4':r4, 'C4h':r4,\
            'D4':r5, 'C4v':r5,'D2d':r5, 'D4h':r5,\
            'C3':r6, 'S6':r6,\
            'D3':r7, 'C3v':r7,'D3d':r7,\
            'C6':r8, 'C3h':r8,'C6h':r8,\
            'D6':r9, 'C6v':r9,'D3h':r9, 'D6h':r9,\
            'T':r10, 'Th':r10,\
            'Td':r11, 'O':r11,'Oh':r11\
        }
        
        if pointGroup not in allowed_Bpars:
            raise ValueError(f'The desired point group name "{pointGroup}" is improper or not implemented')
        
        return allowed_Bpars[pointGroup]
    
    def __init__(self, pointGroup, Bpars, units):
        self.B_names = self.allowed_Bpars(pointGroup)
    
        self.pointGroup = pointGroup
        self.lattice = self._assignLattice(pointGroup)


        # Check units and assign conversion factors
        unitConversions = {'meV':1, 'K':C.K2meV, 'invcm':C.invcm2meV}
        if units not in unitConversions:
            raise ValueError(f'The desired unit "{units}" not in the list of implemented ones: {unitConversions}')
        
        # First, set all to zero
        self.B_values = np.zeros(len(self.B_names))
        
        for it,Bval in enumerate(Bpars):
            self.B_values[it] = Bval*unitConversions[units]
                
        # In case of cubic symmetry the Bpars are given in form B40, B60, B66
        # and some additional symmetry constraints are implemented
        if self.lattice == 'cubic':
            # Althoguh the the cubic symmetry helps in reducing number of parameters it does not with implementation...
            B40 = self.B_values[0]
            B60 = self.B_values[1]
            B66o4 = self.B_values[2]
            
            if pointGroup in ['T','Th']:
                self.B_values[0] = B40
                self.B_values[1] = 5/2*B40
                self.B_values[2] = B60
                self.B_values[3] = -B66o4
                self.B_values[4] = -21/2*B60
                self.B_values[5] = B66o4
            elif pointGroup in ['Td','O','Oh']:
                self.B_values[0] = B40
                self.B_values[1] = 5/2*B40
                self.B_values[2] = B60
                self.B_values[3] = -21/2*B60

            
    def __str__(self):
        ret = f"Set of CEF parameters for {self.pointGroup} point group in a {self.lattice} lattice\n"
        for Bname, Bvalue in zip(self.B_names, self.B_values): 
            ret += f'{Bname} = {Bvalue:.4} meV\n'
            
        return ret
            
    def _assignLattice(self, pointGroup):
        PG2lattice = {
            'C2':'monoclinic', 'Cs':'monoclinic', 'C2h':'monoclinic',\
            'C2v':'orthorhombic','D2':'orthorhombic', 'D2h':'orthorhombic',\
            'C4':'tetragonal', 'S4':'tetragonal', 'C4h':'tetragonal',\
            'D4':'tetragonal', 'C4v':'tetragonal','D2d':'tetragonal', 'D4h':'tetragonal',\
            'C3':'trigonal', 'S6':'trigonal',\
            'D3':'trigonal', 'C3v':'trigonal','D3d':'trigonal',\
            'C6':'hexagonal', 'C3h':'hexagonal','C6h':'hexagonal',\
            'D6':'hexagonal', 'C6v':'hexagonal','D3h':'hexagonal', 'D6h':'hexagonal',\
            'T':'cubic', 'Th':'cubic',\
            'Td':'cubic', 'O':'cubic','Oh':'cubic'\
        }
        
        return PG2lattice[pointGroup]