import numpy as np
import math

from . import Beamline
# import mikibox.crystallography as mscryst

class MLZ_HEIDI(Beamline):
    '''
    High-energy diffractometer
    HEiDi @ MLZ
    Diffractometer
    '''
    
    def __init__(self):
        # I do not know the configuration for HEiDi, leave nominal values
        self._omega_sense = 1
        self._omega_offset = 0
        
        self._gamma_sense = 1
        self._gamma_offset = 0
        
        self._nu_sense = 1
        self._nu_offset = 0

    def import_data(self, filename):
        '''
        Import data from HEiDi funny format.
        '''
        with open(filename,'r') as dataFile:
            lines = dataFile.readlines()
        
        headerLines = 7
        it = headerLines
        n = 1
        
        DATA = {}
        
        while it < len(lines):
            subheader = lines[it]
            h,k,l,_,_,_,_,Npoints,_,_,temperature,_ = subheader.split()
            h,k,l,Npoints,temperature = int(h),int(k),int(l),int(Npoints),float(temperature)
            dataBlockLines = math.ceil(Npoints*2/16)
                    
            dataLines = lines[it+2:it+2+dataBlockLines]
            dataLine = ''.join(dataLines).replace('\n','')
            N = 5
            data = np.array([int(dataLine[i:i+N]) for i in range(0, len(dataLine), N)])
            counts = data[:Npoints]
            monitor = data[Npoints:]
            
            DATA[(h,k,l)] = dict(entry_num=n, temperature=temperature, counts=counts, monitor=monitor)
            
            it += dataBlockLines+2
            n += 1
        
        return DATA

    
    def import_datas(self, ni, nf):
        '''
        Import set of data with numors between ni and nf.
        '''

        PATH = 'C:\\Users\\Michal\\Dropbox\\PostDoc-TUM\\3_CePdAl3\\1_HEIDI\\cpa1\\raw\\cpa1_raw\\'
        
        return {i:self.import_data(f'{PATH}cpa1lt.{i:02d}') for i in range(ni,nf+1)}