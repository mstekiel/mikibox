import numpy as np
import time as tt
from scipy.optimize import curve_fit
from collections import OrderedDict

from . import Beamline

class ILL_D23(Beamline):
    '''
    D23 @ ILL
    Diffractometer
    '''
    
    def __init__(self):
        self.omega_sense = 1
        self.omega_offset = -90
        
        self.gamma_sense = 1
        self.gamma_offset = 0
        
        self.nu_sense = 1
        self.nu_offset = 0
        
    def convert_d23_to_nicos(self, filename: str) -> list[str]:
        '''
        convert_d23_to_nicos(filename)
        Script to convert the D23 format to NICOS dat format, with some tweaks, so that it can be read by Davinci.
        
        Important notes:
            1. omega from D23 needs +90 offset to match conventions of Davinci
            2. A list of header names is appended to be readable by Davinci
            3. There is a warning implemented, that triggers when the incoming or scattered beam might be obstructed by soft pipes.
               They can reduced the beam by 33%. This warning should be tested.
        '''
        
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

    def check_pipes_obstruction(oS:float, oE:float, gamma: float) -> tuple(str, str):
        '''
        check_pipes_obstruction(oS, oE, gamma)
        [oS, oE] : range of the omega scan
        gamma : detector angle

        Check if the soft tubes of the magnet obstruct the beam.
        Gor gamma=nu=0 the tubes occur in the omega range 22-44 deg. and can reduce the intensity by 33%.
        Check possible influence both for the incoming and scattered beam.
        The convention for omega angles is [-180:180].
        '''
        pS, pE = 22, 44
        go = np.sort([gamma-oS, gamma-oE])

        if oS<-180 or oS>180 or oE<-180 or oE>180:
            raise ValueError('Omega values dont make sense while checking obstruction of the incoming beam')

        scattered_check = (-1, '')
        incoming_check = (-1, '')

        # Check scattered beam
        if oE<gamma+pS or oS>gamma+pE:
            scattered_check = (False, f'Pipes dont obstruct the scattered beam')
        else:
            scattered_check = (True, f'Pipes obstruct the scatterred beam in this omega range.')
            
        # Check incoming beam  
        if oS>-180+pE or oE<-180+pS:
            incoming_check = (False, f'Pipes dont obstruct the incoming beam')
        else:
            incoming_check = (True, f'Pipes obstruct the incoming beam in this omega range')

        return scattered_check, incoming_check

    def load_nicos(filename: str) -> tuple:
        '''
        Load datafile of an omega scan from the NICOS dat file format
        '''

        with open(filename, 'r') as ff:
            lines = ff.readlines()

        HEADER = OrderedDict()
        DATA = []

        for line in lines:
            if line[:3] == '#  ':
                key = line[1:27].strip()
                value = line[29:].strip()

                HEADER[key] = value
            elif line[:3] == '###':
                continue
            else:
                om, det, mon1, time = [float(x) for x in line.split()]
                DATA.append([det, mon1, time, om])

        return np.array(DATA), HEADER

    def gauss(self, x,a,x0,sigma,bkg):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))+bkg

    def fit_gauss(self, x: list, y: list, puser: list=False) -> tuple:
        if puser:
            p0 = puser
        else:
            x0 = x[np.argmax(y)]
            sigma = np.abs(x[0]-x[-1])/10
            bkg = y[0]
            
            p0 = [max(y)-bkg,x0,sigma,bkg]

        try:
            popt,pcov = curve_fit(self.gauss,x,y,p0=p0,maxfev=200)
        except RuntimeError:
            popt = np.zeros(4)
            pcov = np.zeros((4,4))

        return popt, pcov

    def check_scantime(filename: str) -> str:
        with open(filename) as ff:
            lines = ff.readlines()
        
        return lines[5].split()[-1]

    def check_scantype(filename: str) -> str:
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
        
    def load_d23(self, filename: str) -> tuple:
        # Read omega scan from native D23 raw format
        
        DATA = []
        HEADER = OrderedDict()
        
        with open(filename) as ff:
            lines = ff.readlines()

        # Read the numor that resides before the main header
        numor = int(lines[1].split()[0])
        HEADER['number'] = numor

        # Read in the header values, pray that they are always in the same lines
        for it in np.arange(22,31+1):
            linechunks1 = [lines[it][i:i+16] for i in range(0,len(lines[it]),16)]
            keys = [str(n).strip() for n in linechunks1]
            vals = [float(n) for n in lines[it+10].split()]

            for key, val in zip(keys,vals):
                HEADER[key] = val

        # Load the data. Number of columns depends on the type of scans, but can be deduced from number of k-points measured and total number of entries
        nentries = int(lines[45].split()[0])
        nkmes = int(lines[16].split()[5])
        
        ncols = int(nentries/nkmes)
        
        for line in lines[46:]:
            DATA = np.append(DATA, [float(n) for n in line.split()])

        try:
            DATA=DATA.reshape((int(nentries/ncols),ncols))
        except BaseException:
            raise ValueError('FILE:',filename,'  Couldnt reshape DATA array to shape given in the header')

        return DATA, HEADER

