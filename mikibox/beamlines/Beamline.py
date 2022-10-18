import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
from ..functions import pseudoVoigt_bkg, gauss_bkg

class Beamline():
    '''
    Class representing a beamline.
    
    It's main purpose is to store the rotation conventions and offsets and avoid retyping.
    Currently implemented things are:
        omega
        gamma
        nu
        
    '''

    def __init__(self, omega_sense=1,omega_offset=0, gamma_sense=1,gamma_offset=0, nu_sense=1,nu_offset=0):
        self._omega_sense = omega_sense
        self._omega_offset = omega_offset
        
        self._gamma_sense = gamma_sense
        self._gamma_offset = gamma_offset
        
        self._nu_sense = nu_sense
        self._nu_offset = nu_offset
            

    def calHKL(self, UB, lbd, omega, gamma, nu):
        '''
        calHKL(UB, lbd, omega, gamma, nu)
        Calculate the current position of the detector in reciprocal space, based on the UB matrix and the real space angles.
        lambda in Angstroems
        angles in degrees
        U matrix convention: orthonormal matrix rotating crystal axes to experimental coordinate system.
        B matrix convention: upper triangle with crystal axes coordinate system in reciprocal space, with a*=1/a lengths
        '''
        omega = np.radians(self.omega_sense*omega + self.omega_offset)
        gamma = np.radians(self.gamma_sense*gamma + self.gamma_offset)
        nu = np.radians(self.nu_sense*nu + self.nu_offset)

        co, so = np.cos(omega), np.sin(omega)
        R = [[co, -so, 0],[so,co,0],[0,0,1]]
        UBm = np.linalg.inv(np.dot(R,UB))
        Qxyz = np.array([np.cos(nu)*np.cos(gamma)-1 , np.cos(nu)*np.sin(gamma), np.sin(nu)])/lbd
        return np.dot(UBm,Qxyz)
        
    def LL_integrate(self, counts, omega_step):
        '''
        Integrate the intensity of a reflection measured in an omega scan by a point detector.
        The method follows the Lehman-Larsen algorithm: Lehmann & Larsen (1974). Acta Cryst. A30, 580-584
        '''
        
        m = np.argmax(counts)
        n = len(counts)
        beta = n//10
        
        def sJJ_right(p):
            sJ = np.sum(counts[m:p+1]) + np.sum(counts[p+1:p+beta+1])*((p-m+1)/beta)**2
            J = np.sum(counts[m:p+1]) - np.sum(counts[p+1:p+beta+1])*((p-m+1)/beta)
            return sJ/J
            
        def sJJ_left(p):
            sJ = np.sum(counts[p:m+1]) + np.sum(counts[p-beta:p-1+1])*((p-m+1)/beta)**2
            J = np.sum(counts[p:m+1]) - np.sum(counts[p-beta:p-1+1])*((p-m+1)/beta)
            return sJ/J
            
      
        sJJ_r = [sJJ_right(p) for p in range(m+3,n-beta)]
        br = np.argmin(sJJ_r)+m+2
            
        sJJ_l = [sJJ_left(p) for p in range(beta,m-3)]
        bl = np.argmin(sJJ_l)+beta
        print(sJJ_l)
        
        return 0, 0, bl, br
        
    def fit_integrate(self, x, y, silent=False):
        # Determine initial parameters
        x0 = x[np.argmax(y)]
        s = 2*np.abs(x[1]-x[0])
        A = max(y)
        bkg = (y[0]+y[-1])/2
        eta = 0.5
        pinit = (x0,A,s,eta,bkg)
        
        # Fit gaussian
        popt, pcov = curve_fit(pseudoVoigt_bkg, x,y, p0=pinit, bounds=([x[0],0,s,0,-np.inf],[x[-1],np.inf,(x[-1]-x[0])/2,1,np.inf]))
        
        bkg_integral = popt[-1]*np.abs(x[0]-x[-1])
        
        I_integrated = integrate.simpson(y,x)-bkg_integral
        I_fit = popt[1]*np.sqrt(2*np.pi)*popt[2]
        
        if np.abs((I_integrated-I_fit)/I_integrated) > 0.1 and  not silent:
            raise Warning('Interated intensity inconsistent with the fit one.')
        
        
        return I_integrated, popt
