import numpy as np
import time as tt

from .Beamline import Beamline
from ..crystallography import *

class PSI_ZEBRA(Beamline):
    '''
    ZEBRA @ PSI
    Diffractometer
    '''
    
    def __init__(self):
        self.omega_sense = -1
        self.omega_offset = -90
        
        self.gamma_sense = -1
        self.gamma_offset = 0
        
        self.nu_sense = 1
        self.nu_offset = 0
    
    def integrate_dataset(self, DATA, normalize=True, scale=100):
        '''
        Integrate the dataset.
        
        Return
            List containing entries as in the hkl file format
        '''
        list_hkl = []
        for DATA_block in DATA:
            h,k,l = [DATA_block[key] for key in 'hkl']
            II = self.fit_integrate(DATA_block['x'], DATA_block['counts'])
            dII = np.sqrt(II)
            
            # Normalize to monitor
            if normalize:
                II = scale*II/DATA_block['monitor']
                dII = scale*dII/DATA_block['monitor']
                
            list_hkl.append([h,k,l,II,dII])
            
        return list_khl
        
    def get_refl(self, DATA,hkl):
        '''
        Get the datablock corresponding to the reflection
        '''
        h,k,l= hkl
        
        refl_id=-1
        for it,DATAblock in enumerate(DATA):
            if DATAblock['h']==h and DATAblock['k']==k and DATAblock['l']==l:
                refl_id = it
                
        if refl_id==-1:
            raise Warning(f'Reflection {hkl} not found in the dataset')
            return None
            
        return DATA[refl_id]
        
    def load_ccl(self, filename, **kwargs):
        '''
        Read the ccl file containing multiple scans of various reflections
        
        Some additional entries regarding the scan are included in the header.
        
        Returns:
            HEADER : dictionary
                Contains the main header of the ccl file
            DATA : array of dictionaries
                Each dictionary contains a measured reflection. Main data is stored in DATA[n]['counts'] and supplementary DATA[n]['x'], 
                where x is omega, or omega-twotheta for high Q.
        '''
        
        # Handle kwargs
        if 'block_start_offset' in kwargs:
            block_start_offset = kwargs['block_start_offset']
        else:
            block_start_offset = 0
            
        DATA = []
        HEADER = {}
        
        with open(filename) as ff:
            lines = ff.readlines()

        data_start = -1     
            
        for it,line in enumerate(lines):
            if len(line)>5 and '=' in line:
                key, value = [record.rstrip() for record in line.split('=')]
                
                try:
                    value = float(value)
                except ValueError:
                    pass
                    
                HEADER[key] = value
                
            if line[:2]=='UB':
                pass
                
            if line=='#data\n':
                data_start = it
                
        # Read Npoints for a reflection. It should be the same for all reflections in a single ccl file.
        block_start = data_start+1+block_start_offset
        Npoints = int(lines[data_start+2+block_start_offset].split()[0])
        block_lines = int(Npoints//10+1+2)
        
        while block_start < len(lines):
            # Read block header
            keys1 = ['Nblock', 'h', 'k', 'l', 'tth', 'om', 'nu', '_', '_']
            vals1 = [float(x) for x in lines[block_start].split()]
            keys2  =['Npoints', 'step', 'monitor', 'temperature', 'magnetic_field', 'date', 'time', 'scantype', 'step2']
            vals2 = [x if it in [5,6,7] else float(x) for it,x in enumerate(lines[block_start+1].split())]
            
            # Assign block header
            DATA_subset = {}
            for key,val in zip(keys1,vals1):
                DATA_subset[key] = val
                
            for key,val in zip(keys2,vals2):
                DATA_subset[key] = val
                
            # Read counts
            counts_start = block_start+2
            counts_end = counts_start + block_lines - 2
            counts = np.array([int(cts) for cts in ' '.join(lines[counts_start:counts_end]).split()])
            
            # Asign counts and additional entries
            DATA_subset['counts'] = counts
            x_start = DATA_subset['om'] - Npoints/2*DATA_subset['step']
            #x_end = DATA_subset['om'] + Npoints/2*DATA_subset['step']
            DATA_subset['x'] = x_start+np.arange(0,Npoints,1)*DATA_subset['step']
                
            
            DATA.append(DATA_subset)
            
            block_start += block_lines
        
        return HEADER, DATA

    def load_dat(self, filename):
        '''
        Read the dat file containing a single scan.
        
        Some additional entries regarding the scan are included in the header.
        '''
        DATA = []
        HEADER = {}
        
        with open(filename) as ff:
            lines = ff.readlines()
            
        data_start = -1   
        data_end = -1    
            
        for it,line in enumerate(lines):
            line_records = line.split()
            if len(line_records)>1 and line_records[1]=='=':
                it = line.find('=')
                
                try:
                    value = float(line[it+2:])
                except ValueError:
                    value = line[it+2:]
                    
                HEADER[line_records[0]] = value
                
            if line=='#data\n':
                data_start = it
                
            if line=='END-OF-DATA\n':
                data_end = it
                
        data_points = int(lines[data_start+2].split()[0])
        HEADER['data_points'] = data_points
        
        data_header = lines[data_start+3].split()
        HEADER['data_header'] = data_header
        
        # This does not work for qscans so lets omit it
        # scan_line = lines[data_start+1]
        # trimmed_line = scan_line.replace('Scanning Variables:','').replace('Steps:','')
        # scan_variables, scan_steps = trimmed_line.split(',')
        # HEADER['scan_variables'] = scan_variables.replace(' ','')
        # HEADER['scan_steps'] = [float(x) for x in scan_steps.split()]
        
        
        DATA_str = [x.split() for x in lines[data_start+4:data_end]]
        DATA = np.array(DATA_str, dtype=float)

                
        return DATA, HEADER
        
    def load(self,filename, extension=None):
        '''
        Wrapper for loading one of the two filetypes used at ZEBRA
        '''

        # Figure out the format from file extension
        if extension == None:
            if filename[-3:] == 'ccl':
                extension = 'ccl'
            elif filename[-3:] == 'dat':
                extension = 'dat'
            else:
                raise ValueError('Unknown extension of the file')

                
        if extension == 'ccl':
            DATA, HEADER = load_ccl(filename)
        elif extension == 'dat':
            DATA, HEADER = load_dat(filename)
        else:
            raise ValueError('Unknown extension of the file')   
            
        return DATA, HEADER
        
    def convert_CCL2NICOS(self, filename, **kwargs):
        '''
        Convert the packed ccl file with measured reflections to the NICOS dat format.
        
        A Strange offset of omega=-135 deg. has to be applied in order to be read properly by davinci
        
        Returns:
            List of strings, where each element is a NICOS file
        '''
        
        HEADER, DATA = self.load_ccl(filename, **kwargs)

        outfiles = []

        for DATA_block in DATA:
            outlines = []

            outlines.append('### NICOS data file. Created at %s' % (tt.strftime('%Y-%m-%d %H:%M:%S',tt.localtime())))

            # Write up the header
            outlines.append('### Header copied from the main header of the ZEBRA CCL file')
            for key in HEADER:
                outlines.append('#%26s : %s' % (key, HEADER[key]))

            # Some other interesting values derived from the header
            outlines.append('### Interesting values, derived from the header')
        
            UB = HEADER['UB'].replace('(','').replace(')','').replace(',','')
            UB = np.reshape(np.array(UB.split(),dtype=float), (3,3))
                    
            lengths, angles = lattice_pars_from_UB(UB)
            outlines.append('#%26s : %s A' % ('lattice (a,b,c))', lengths))
            outlines.append('#%26s : %s deg' % ('lattice (alp,bet,gam))', angles))

            # Some header names need to be added to be readable by Davinci
            outlines.append('### Davinci readable values')
            
            outlines.append('#%26s : %s' % ('Sample_ubmatrix', str(UB).replace('\n',' , ')))
            #outlines.append('#%26s : %s deg' % ('sth_value', self.omega_sense*DATA_block['om']+self.omega_offset))
            outlines.append('#%26s : %s deg' % ('chi2_value', '0.00'))
            outlines.append('#%26s : %s K' % ('Ts_value', DATA_block['temperature']))
            outlines.append('#%26s : %s T' % ('B_value', DATA_block['magnetic_field']))
            outlines.append('#%26s : %s A' % ('wavelength_value', HEADER['wavelength']))
            outlines.append('#%26s : %s deg' % ('omega_value', self.omega_sense*DATA_block['om']+self.omega_offset))
            outlines.append('#%26s : %s deg' % ('gamma_value', self.gamma_sense*DATA_block['tth']+self.gamma_offset))
            outlines.append('#%26s : %s deg' % ('chi1_value', '0.00'))
            outlines.append('#%26s : %s deg' % ('liftingctr_value', self.nu_sense*DATA_block['nu']+self.nu_offset))

            outlines.append('### Scan data')
            outlines.append('# %6s %8s %8s' % ('omega', 'ctr1', 'mon1'))
            outlines.append('# %6s %8s %8s' % ('deg', 'cts', 'cts'))
            
            om, cts, monitor = [DATA_block[key] for key in ['x','counts','monitor']]
            for om_it, cts_it in zip(om, cts):
                outlines.append('%8.2f %8d %8d' % (self.omega_sense*om_it+self.omega_offset, cts_it, monitor))

            outlines.append('### End of NICOS data file')
            
            outfiles.append(outlines)
            
            #print(self.calHKL(UB,HEADER['wavelength'],DATA_block['om'],DATA_block['tth'],DATA_block['nu']))

        return outfiles