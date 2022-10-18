import numpy as np
import matplotlib as mpl

import mikibox as ms

from .lattice import Lattice
from ..beamlines import Beamline

class AbsorptionCorrection():
    '''
    Object representing the problem of absorption correction.
            
    Attributes:
        Lattice: mikibox.crystallography.Lattice
            Lattice object.
        Beamline: mikibox.beamlines.Bemaline
            Beamline object to get the rotation conventions and senses of the used elements.
        absrption_coefficient: float
            Number representing the lienar absorption coefficient in 1/mm units. Satisfies the equation :math:`I(d)=I_0 exp(-\\mu d)`,
            where :math:`I_0` is th incoming flux, and :math:`d` is the thickness.
        smpmle_shape: 'cuboid', 'spherical'
            Type of the sample shape.
        sample_dimensions: ndarray(float)
            Array describing the dimensions of the sample. Its length depends on the `sample_shape`.

    '''

    def __init__(self, lattice:Lattice, beamline:Beamline, absorption_coefficient:float, sample_shape:str, \
                sample_dimensions:tuple, sample_dimensions_reference:tuple):
        # Copy the beamline field
        self.beamline = beamline

        # Copy the lattice field, but also update the orientation matrix to the 
        # `sample_dimensions_reference` coordinate system.
        self.lattice = lattice
        self.lattice.updateOrientation(sample_dimensions_reference)
        
        # Check validity of the parameters
        if sample_shape not in ['cuboid', 'spherical']:
            raise IndexError('Unknown sample shape')
            
        # Now fill the other fields
        self.absorption_coefficient = absorption_coefficient
        self.sample_shape = sample_shape
        self.sample_dimensions = sample_dimensions
        self.sample_dimensions_reference = sample_dimensions_reference
        
        
    def __str__(self):
        return str(self.lattice_parameters)
        
    def path_central(self, phi):
        '''
        Calculate the path within the sample based on the central element method only.
        '''
        
        if self.sample_shape=='cuboid':
            d1, d2 = self.sample_dimensions
            alpha = np.arctan2(d2/2,d1/2)
            phi2 = np.pi/2+phi
            
            if np.abs(np.cos(phi)) > np.cos(alpha):
                l = d1/2/np.abs(np.cos(phi))
            else:
                l = d2/2/np.abs(np.sin(phi))
        elif self.sample_shape=='spherical':
            l = self.sample_dimensions
                
            
        return l
        
    def rotate_to_diffract(self, hkl, wavelength):
        '''
        Calculate the angle necessary to put the sample in diffraction condition.
        '''
        
        assert self.lattice.is_in_scattering_plane(hkl)
        
        tth = self.lattice.scattering_angle(hkl, wavelength)
        
        n1 = self.lattice.hkl2Q(self.sample_dimensions_reference[0])
        n2 = self.lattice.hkl2Q(self.sample_dimensions_reference[1])
        Qhkl = self.lattice.hkl2Q(hkl)
        
        alpha1 =  ms.angle(n1, Qhkl)*np.sign(-Qhkl[1])
        psi = (tth + (np.pi-tth)/2)*self.beamline._omega_sense
        
        return alpha1+psi
        
    def attenuation(self, hkl_list, wavelength):
        '''
        Calculate the attenuation of the incoming and scattered beam for the given (hkl).
        '''
        
        attenuation_list = []
        for hkl in hkl_list:
            tth = self.lattice.scattering_angle(hkl, wavelength)
            n1 = self.lattice.hkl2Q(self.sample_dimensions_reference[0])
            Qhkl = self.lattice.hkl2Q(hkl)
            phi = tth/2 - ms.angle(n1, Qhkl)
            
            l_ki = self.path_central(phi)
            l_kf = self.path_central(-2*phi)
        
            attenuation = np.exp(-self.absorption_coefficient*(l_ki+l_kf))
            
            attenuation_list.append(attenuation)
            
        return attenuation_list
        
    def plot_beam_path(self, ax, hkl, wavelength):
        '''
        Inspection tool to plot the sample, its orientation and the incoming, scattered beam and the chosen hkl.
        '''
        
        d1, d2 = self.sample_dimensions
        klength = 1.5*max(d1,d2)
        tth = self.lattice.scattering_angle(hkl, wavelength)
        
        arrow_styles = dict(width=klength/100, head_width=klength/20, length_includes_head=True, zorder=0)
        sample_style = dict(ec='black', fc='gray', alpha=0.3, zorder=-20)

        # Sample      
        psi = self.rotate_to_diffract(hkl, wavelength)
        print(psi, np.degrees(psi))
        
        # Show sample_simensions_reference arrows
        if False:
            n1 = 5*self.lattice.hkl2Q(self.sample_dimensions_reference[0])
            n2 = 5*self.lattice.hkl2Q(self.sample_dimensions_reference[1])
            n1_rot = np.dot(ms.Rz(psi), n1)
            n2_rot = np.dot(ms.Rz(psi), n2)
            ax.arrow(x=0,y=0,dx=n1_rot[0],dy=n1_rot[1], color='tab:gray', **arrow_styles)
            ax.arrow(x=0,y=0,dx=n2_rot[0],dy=n2_rot[1], color='tab:gray', **arrow_styles)

        l_ki = self.path_central(psi)
        l_kf = self.path_central(psi+np.pi-tth)
        
        sample = mpl.patches.Rectangle(xy=(-d1/2,-d2/2), width=d1, height=d2, angle=np.degrees(psi), rotation_point='center', **sample_style)
        ax.add_artist(sample)
        
        # ki
        ki = np.array([klength,0])
        ax.arrow(x=-klength,y=0,dx=ki[0],dy=ki[1], color='tab:red', **arrow_styles)
        ax.annotate(f'k$_i$ l$_i$={l_ki:.2f} mm', xy=(-klength/2,klength/20), color='tab:red')
        
        # kf
        kf = klength*np.array([np.cos(tth), self.beamline._omega_sense*np.sin(tth)])
        ax.arrow(x=0,y=0,dx=kf[0],dy=kf[1], color='tab:red', **arrow_styles)
        ax.annotate(f'k$_f$ l$_f$={l_kf:.2f} mm', xy=(kf[0]/2+klength/20, kf[1]/2), color='tab:red')

        # Q
        Q = kf-ki
        ax.arrow(x=0,y=0,dx=Q[0],dy=Q[1], color='tab:purple', **arrow_styles)
        ax.annotate(f'Q$_{{{hkl}}}$', xy=Q, color='tab:purple')
        
        return