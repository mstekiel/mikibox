import numpy as np

from .lattice import Lattice


class HKLdata():
    def __init__(self, filename=None, handling_flag=None):
        '''
        Load the data of the hkl file into internal structure.

        Types of hkl files, described by `handling_flag`
            None
                Self made that dont finish with (000) reflection and only consist of data.
            'proper'
                Proper ones that finish with (000) reflection
            'crysalis'
                Crysalis ones that finish with (000) reflection and heve ins file input at the end
        '''

        if isinstance(filename, None):
            # Blank instance of the class
            self.data = None
        else:
            if isinstance(handling_flag, None):
                # Self made that dont finish with (000) reflection and only consist of data.
                self.data = np.loadtxt(filename)
            elif handling_flag=='proper':
                # Proper ones that finish with (000) reflection

                self.data = np.loadtxt(filename)

                # Delete the last entry with (000) reflection
                del self.data[-1]
            elif handling_flag=='crysalis':
                # Crysalis ones that finish with (000) reflection and heve ins file input at the end

                with open(filename, 'r') as ff:
                    lines = ff.readlines()

                    self.data = []

                    for line in lines:
                        h,k,l,I,dI,flag = [float(x) for x in line.split()]

                        self.data.append(np.array([]))
                    
            

