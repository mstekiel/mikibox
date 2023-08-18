import numpy as np
import time as tt
from scipy.optimize import curve_fit
from collections import OrderedDict

from . import Beamline
from .. import physics

class HFIR_HB3(Beamline):
    '''
    HB3 @ HFIR
    Threeaxis spectrometer
    '''
    
    def __init__(self):
        self.omega_sense = 1
        self.omega_offset = -90
        
        self.gamma_sense = 1
        self.gamma_offset = 0
        
        self.nu_sense = 1
        self.nu_offset = 0

    def load_data(self, filename: str):
        '''
        Load the data into (HEADER, DATA)
        '''

        DATA = []
        HEADER = {}

        with open(filename, 'r') as ff:
            lines = ff.readlines()

        # Figure out ranges
        it_data_lines = [it for it,line in enumerate(lines) if line[0]!='#']
        it_data_header = it_data_lines[0]-1
        it_header_lines = np.arange(0, it_data_lines[0]-1)

        # Prepare HEADER
        for it_header_line in it_header_lines:
            line = lines[it_header_line]

            it = line.find('=')
            key = line[1:it].strip()    # skip the initial #
            value = line[it+1:].strip()
            HEADER[key] = value

        # Prepare data_type for the DATA
        col_names = lines[it_data_header].split()[1:]   # Skip initial #
        cur_dtype = np.dtype({'names':col_names, 'formats':['f4' for _ in range(len(col_names))]})

        data_str = [tuple(lines[it].split()) for it in it_data_lines]
        DATA = np.array(data_str, dtype=cur_dtype)

        return HEADER, DATA
    
    def scale_counts(self, counts: np.ndarray, monitor: np.ndarray, new_monitor: float, scaling_type: str) -> tuple[np.ndarray, np.ndarray]:
        '''
        Scale counts to the monitor value.
        '''
        if scaling_type=='lin':
            z = counts/monitor*new_monitor
            z_error = np.sqrt(counts)/monitor*new_monitor
        elif scaling_type=='log':
            z = np.log1p(counts/monitor*new_monitor)
            z_error = np.log1p(np.sqrt(counts)/monitor*new_monitor)
        elif scaling_type=='raw':
            z = counts
            z_error = np.sqrt(counts)
        else:
            raise KeyError(f'Improper count scaling chosen: {scaling_type}')
        
        return z, z_error
    
    def rescale_bose(self, energies: np.ndarray, counts: np.ndarray, temperature_old: float, temperature_new: float) -> tuple[np.ndarray, np.ndarray]:
        '''
        Rescale the counts acccording to Bose occupation factor.
        Usefule when rescaling phonons measured at `old_tempreature` to compare with data at `new_temperature`.

        Will multiply `counts` array by math:`n(E, T_{new})/n(E, T_{old})`, where:
        math:`n(E, T) = 1/(exp(E/k_B T)-1)`

        Parameters:
            energies:
                Energy transfer array.
            counts:
                Intensity measured ant each energy transfer point.
            temperature_old:
                Temperature at which the data was originally measured.
            temperature_new:
                Temperature to which rescale the counts.
        '''

        assert energies.shape == counts.shape

        bose_refactor = physics.bose_occupation(energies, temperature_new) / physics.bose_occupation(energies, temperature_old)
        print(bose_refactor)
        
        return counts*bose_refactor
    

    ########### Old, copied

    # That doesnt work for non np.array datatypes
    def save_to_np(self, np_filename: str, numors: list, path: str, scan_type: list[str]=['any']) -> None:
        '''
        Load scans from given `numors` and save in a `np_filename`.
        Only scans of type `scan_type` will be loaded, that should match the 'def_x' variable of the header.
        '''

        ALL_SCANS = {}

        for numor in numors:
            filename = f'{path}/HB3_exp0769_scan{numor:04}.dat'

            HEADER, DATA = self.load_data(filename)

            if HEADER['def_x'] in scan_type or scan_type[0]=='any':
                ALL_SCANS[numor] = [HEADER, DATA]

        np.save(np_filename, ALL_SCANS)

        return


        
    def convert_to_nicos(self, filename: str) -> list[str]:
        '''
        Convert the ASCII file format of IN12 to Nicos
        '''

        return []
        
        outlines = []

        data, header = self.load_d23(filename)

        outlines.append('### NICOS data file. Created at %s' % (tt.strftime('%Y-%m-%d %H:%M:%S',tt.localtime())))

        # Write up the header
        outlines.append('### Header copied from D23 file format')
        for key in header:
            outlines.append('#%26s : %s' % (key, header[key]))

        # Some other interesting values derived from D23 header
        outlines.append('### Interesting values, derived from D23 header')

        UB = self.getUB(filename)
        lengths, angles = self.lattice_pars_from_UB(UB)
        outlines.append('#%26s : %s A' % ('lattice (a,b,c))', lengths))
        outlines.append('#%26s : %s deg' % ('lattice (alp,bet,gam))', angles))

        # Some header names need to be added to be readable by Davinci
        outlines.append('### Davinci readable values')
        outlines.append('#%26s : %s' % ('Sample_ubmatrix', str(UB).replace('\n',' , ')))
        outlines.append('#%26s : %s deg' % ('sth_value', header['scan start']+header['scan width']/2.))
        outlines.append('#%26s : %s deg' % ('chi2_value', '0.00'))
        outlines.append('#%26s : %s K' % ('Ts_value', header['Temp-sample']))
        outlines.append('#%26s : %s T' % ('B_value', header['Mag.field']))
        outlines.append('#%26s : %s A' % ('wavelength_value', header['wavelength']))
        outlines.append('#%26s : %s deg' % ('omega_value', header['scan start']+header['scan width']/2.))
        outlines.append('#%26s : %s deg' % ('gamma_value', header['2theta (gamma)']))
        outlines.append('#%26s : %s deg' % ('chi1_value', '0.00'))
        outlines.append('#%26s : %s deg' % ('liftingctr_value', header['chi']))

        # Soft pipes warning
        kf_check, ki_check = self.check_pipes_obstruction(header['scan start'], header['scan start']+header['scan width'], header['2theta (gamma)'])

        outlines.append('### Magnet pipes check')
        outlines.append('#%26s : %s' % ('Incoming beam obstructed', ki_check))
        outlines.append('#%26s : %s' % ('Scattered beam obstructed', kf_check))

        outlines.append('### Scan data')
        outlines.append('# %6s %8s %8s %8s' % ('omega', 'ctr1', 'mon1', 'time'))
        outlines.append('# %6s %8s %8s %8s' % ('deg', 'cts', 'cts', 'ms'))
        
        for cts, mon, time, om in data:
            outlines.append('%8.2f %8d %8d %8d' % (om+90, cts, mon, time))

        outlines.append('### End of NICOS data file')
        
        return outlines

    def check_scantime(self, filename: str) -> str:
        with open(filename) as ff:
            lines = ff.readlines()
        
        return lines[5].split()[-1]

    def check_scantype(self, filename: str) -> str:
        with open(filename) as ff:
            lines = ff.readlines()

        return lines[9].split()[-1]
        
    def getUB(self, filename: str) -> np.ndarray:
        '''
        Read the UB matrix directly from the raw data file, without caring about anything else
        '''
        with open(filename) as ff:
            lines = ff.readlines()
            
        u11 = float(lines[33].split()[3])
        u12 = float(lines[33].split()[4])
        u13 = float(lines[34].split()[0])
        u21 = float(lines[34].split()[1])
        u22 = float(lines[34].split()[2])
        u23 = float(lines[34].split()[3])
        u31 = float(lines[34].split()[4])
        u32 = float(lines[35].split()[0])
        u33 = float(lines[35].split()[1])
        
        return np.array([[u11,u12,u13],[u21,u22,u23],[u31,u32,u33]])

    def lattice_pars_from_UB(self, UB: np.ndarray) -> tuple:
        U, B = np.linalg.qr(UB)

        a1,a2,a3 = np.linalg.inv(B)

        a = np.sqrt(np.dot(a1,a1))
        b = np.sqrt(np.dot(a2,a2))
        c = np.sqrt(np.dot(a3,a3))

        alp = np.degrees(np.arccos(np.dot(a2,a3)/(b*c)))
        bet = np.degrees(np.arccos(np.dot(a1,a3)/(a*c)))
        gam = np.degrees(np.arccos(np.dot(a1,a2)/(a*b)))

        return ((a,b,c), (alp,bet,gam))
    
# if __name__ == '__main__':
#     filename = 'G:/My Drive/Postdoc-TUM/Projects/6_CSO-Tobias/2_HB3-TAX/oncat-data/Datafiles/HB3_exp0769_scan0050.dat'

#     TAX = HFIR_HB3()

#     TAX.load_data()