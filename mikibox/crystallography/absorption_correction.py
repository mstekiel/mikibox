import numpy as np
import matplotlib as mpl
import copy

from typing import Any

import mikibox as ms

from . import Lattice
from mikibox.instruments import Beamline

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
        smpmle_shape: 'rectangle', 'circle'
            Projection of the sample shape on the scattering plane.
        sample_dimensions: ndarray(float)
            Array describing the dimensions of the sample. Its length depends on the `sample_shape`.

    '''

    def __init__(self, lattice:Lattice, beamline:Beamline, absorption_coefficient:float, sample_shape:str, \
                sample_dimensions: Any, sample_dimensions_reference: Any):
        # Copy the beamline field
        self.beamline = beamline

        # Copy the lattice field, but also update the orientation matrix to the 
        # `sample_dimensions_reference` coordinate system.
        self.lattice = Lattice(lattice.lattice_parameters, orientation=None)
        self.lattice.updateOrientation(sample_dimensions_reference)
        
        # Check validity of the parameters
        assert sample_shape in ['rectangle', 'circle'], 'Non-implemented sample shape requested'
            
        # Now fill the other fields
        self.absorption_coefficient = absorption_coefficient
        self.sample_shape = sample_shape
        self.sample_dimensions = sample_dimensions
        self.sample_dimensions_reference = sample_dimensions_reference
        
        
    def __str__(self):
        return str(self.lattice_parameters)

    #
    # Helper functions
    #

    def rotate_to_diffract(self, hkl: tuple, wavelength: float) -> float:
        '''
        Calculate the angle necessary to put the sample in diffraction condition.
        '''
        
        assert self.lattice.is_in_scattering_plane(hkl)
        
        tth = self.lattice.scattering_angle(hkl, wavelength)
        
        n1 = self.lattice.hkl2Q(self.sample_dimensions_reference[0])
        Qhkl = self.lattice.hkl2Q(hkl)
        
        beta =  ms.angle(n1, Qhkl)*np.sign(-Qhkl[1])
        psi = (tth + (np.pi-tth)/2)*self.beamline._omega_sense
        
        return beta+psi

    #
    # Functions used to calculate attenuation
    # 
        
    def rectangle_radius(self, rectangle_dimensions: tuple, psi: float) -> float:
        r'''
        Calculate the 'radius' of the rectangle as a function of polar angle psi.

        Formulas derived on paper `Docs\\Absorption_paths.pdf`

        rectangle_dimensions: tuple
            Tuple representing (width, height).
        psi: float
            rotation angle in radians.
        '''
        
        lx, ly = rectangle_dimensions
        alpha = np.arctan2(ly/2,lx/2)

        if np.abs(np.cos(psi)) > np.cos(alpha):
            l = lx/2/np.abs(np.cos(psi))
        else:
            l = ly/2/np.abs(np.sin(psi))                
            
        return l

    def polygon_ray(self, points: list) -> float:
        '''
        Determine the length of a line segment [-inf, 0]->[0,0] within a convex polygon
        with vertices defined by `points`.

        Parameters:
            points: list
                List of points definig the polygon. Shape (n, 3), so defined in 3D space.
        '''
        # Calculate the polar angles for all points
        angles = [np.arctan2(p[1], p[0]) for p in points]

        # Find the characteristic points
        p_min = points[np.argmin(angles)]
        p_max = points[np.argmax(angles)]
        
        # Find the line connecting these two points,
        # but take into account that it could by vertical.
        length = None
        if p_min[0]==p_max[0]:
            length = p_min[0]
        else:
            a = (p_min[1]-p_max[1])/(p_min[0]-p_max[0])
            b = p_min[1] - a*p_min[0]
            
            length = -b/a
        
        return np.abs(length)

    def rectangle_element_radius(self, rectangle_dimensions: tuple, element_displacement: tuple, psi: float) -> float:
        r'''
        Calculate the path of the incoming beam reaching the element of a rectangle sample.

        Polygon approach
        
        rectangle_dimensions: tuple
            Tuple representing (width, height).
        element_displacement: tuple
            Displacement of the element with respect to the center.
        psi: float
            rotation angle in radians.
        '''

        # Define the rotated vertices, with respect to the displaced element,
        # that also needs to be rotated
        lx, ly = rectangle_dimensions
        dx, dy = element_displacement

        assert -lx/2 < dx < lx/2, 'elements x-coordinate not within the sample'
        assert -ly/2 < dy < ly/2, 'elements y-coordinate not within the sample'

        p0 = np.dot(ms.Rz(psi), [ dx, dy, 0])
        p1 = np.dot(ms.Rz(psi), [ lx/2, ly/2, 0])
        p2 = np.dot(ms.Rz(psi), [-lx/2, ly/2, 0])
        p3 = np.dot(ms.Rz(psi), [-lx/2,-ly/2, 0])
        p4 = np.dot(ms.Rz(psi), [ lx/2,-ly/2, 0])
        points = np.array([p1, p2, p3, p4]) - p0

        return self.polygon_ray(points)
       

    def displaced_paths(self, hkl: tuple, wavelength: float, method_kwargs: dict=dict()) -> list:
        '''
        Calculate the path for the element within the sample.
        '''
        # Look into the requested options
        Nx = method_kwargs.setdefault('Nx', 9)
        Ny = method_kwargs.setdefault('Ny', 9)
        silent = method_kwargs.setdefault('silent', True)

        l_ki_list, l_kf_list = [], []

        if self.sample_shape == 'rectangle':
            d1, d2 = self.sample_dimensions
            psi = self.rotate_to_diffract(hkl, wavelength)
            tth = self.lattice.scattering_angle(hkl, wavelength)


            sdx, sdy = d1/Nx/2, d2/Ny/2
            it, Ntot = 0, Nx*Ny
            for dx in np.linspace(-d1/2+sdx, d1/2-sdx, Nx):
                for dy in np.linspace(-d2/2+sdy, d2/2-sdy, Ny):
                    if not silent:
                        it += 1
                        print(f'Working on the element {it}/{Ntot}')

                    l_ki_list.append(self.rectangle_element_radius(self.sample_dimensions, (dx, dy), psi))
                    l_kf_list.append(self.rectangle_element_radius(self.sample_dimensions, (dx, dy), psi+(np.pi-tth)*self.beamline._omega_sense))
        else:
            raise KeyError('Sample shape not implemented for the displaced_paths method.')

        return np.array(l_ki_list), np.array(l_kf_list)
        
    def central_paths(self, hkl: tuple, wavelength: float) -> list:
        '''
        Calculate the beam path length in the sample oriented with hkl
        in the diffraction condition.

        hkl: tuple
            Miller indices of the considered reflection.
        wavelength: float
            Wavelength

        Returns:
            l_i, l_f
                Path lengths of the incoming and scattered beams.
        '''

        if self.sample_shape == 'rectangle':
            tth = self.lattice.scattering_angle(hkl, wavelength)
            psi = self.rotate_to_diffract(hkl, wavelength)

            l_ki = self.rectangle_radius(self.sample_dimensions, psi)
            l_kf = self.rectangle_radius(self.sample_dimensions, psi+(np.pi-tth)*self.beamline._omega_sense)
        elif self.sample_shape == 'circle':
            l_ki = l_kf = self.sample_dimensions

        return l_ki, l_kf


    def attenuation(self, hkl: tuple, wavelength: float, method: str='central', method_kwargs: dict=dict()) -> list:
        '''
        Calculate the attenuation of the incoming and scattered beam for the given (hkl).
        It follows the Beer-Lambert's law of exponential decrease of the beam intensity within the sample.
        TODO formula

        hkl : tuple
            Miller indices of the reflection to compute the attenuation for.
        wavelength : float
            Wavelength
        method : 'central', 'grid'
            Method by which to calculate the beam attenuation.
        method_kw: dictionary
            Non-standard specifications for the method used.
            | central -> None
            | grid -> {'Nx':9, 'Ny':9, 'silent':True}
        '''
        

        if method == 'central':
            l_ki, l_kf = self.central_paths(hkl, wavelength)

            attenuation = np.exp(-self.absorption_coefficient*(l_ki+l_kf))
        elif method == 'grid':
            # Each contribution must be weighted by the number of elements the sample was divided in.
            # This number is a length of the list of ki's or kf's

            l_ki_list, l_kf_list = self.displaced_paths(hkl, wavelength, method_kwargs)

            attenuation = np.sum( np.exp(-self.absorption_coefficient*(l_ki_list+l_kf_list)) )/len(l_ki_list)


        else:
            raise ValueError('Invalid method chosen for attenuation evaluation')

            
        return attenuation

    def correct_data_hkl(self, data_hkl: list, wavelength: float, method: str='central', method_kwargs: dict=dict()) -> list:
        '''
        Correct the intensities and errors of data loaded from the hkl file.

        data_hkl: list
            List with shape (5, N), where each row consists of [h,k,l,I,dI].
        absorption_coefficient: float, None
            The mu paramater. If None, the absorption coefficient instantiated with the class initializer will be taken.
        wavelength: float
            Wavelength of the incoming radiation
        method: {'central', 'grid'}
            Method used to calculate beam attenuation.
        method_kwargs: dict
            Additional arguments passed to the method calculating attenuation.

        Returns:
            Copy of the data_hkl with corrected intensities and errors.
        '''

        attenuation = lambda h,k,l : self.attenuation((h,k,l), wavelength, method, method_kwargs)
        data_hkl_corrected = [[h,k,l, I/attenuation(h,k,l), dI/attenuation(h,k,l)] for h,k,l,I,dI in data_hkl]

        return data_hkl_corrected
        
    #
    # Inspection methods
    #

    def plot_beam_path(self, ax: Any, hkl: tuple, wavelength: float) -> None:
        '''
        Inspection tool to plot the sample, its orientation and the incoming, scattered beam and the chosen hkl.
        '''
        
        d1, d2 = self.sample_dimensions
        klength = 1.5*max(d1,d2)
        tth = self.lattice.scattering_angle(hkl, wavelength)
        
        arrow_styles = dict(width=klength/100, head_width=klength/20, length_includes_head=True, zorder=0)
        sample_style = dict(ec='black', fc='gray', alpha=0.3, zorder=-20)

        #
        # Sample      
        #
        psi = self.rotate_to_diffract(hkl, wavelength)
        
        # Show sample_simensions_reference arrows
        if False:
            n1 = 5*self.lattice.hkl2Q(self.sample_dimensions_reference[0])
            n2 = 5*self.lattice.hkl2Q(self.sample_dimensions_reference[1])
            n1_rot = np.dot(ms.Rz(psi), n1)
            n2_rot = np.dot(ms.Rz(psi), n2)
            ax.arrow(x=0,y=0,dx=n1_rot[0],dy=n1_rot[1], color='tab:gray', **arrow_styles)
            ax.arrow(x=0,y=0,dx=n2_rot[0],dy=n2_rot[1], color='tab:gray', **arrow_styles)

        l_ki, l_kf = self.central_paths(hkl, wavelength)
        
        rectangle_shift = np.dot(ms.Rz(psi), [d1/2, d2/2, 0])
        sample = mpl.patches.Rectangle(xy=(-rectangle_shift[0], -rectangle_shift[1]), width=d1, height=d2, angle=np.degrees(psi), **sample_style)
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

    def plot_beam_paths_grid(self, ax: Any, hkl: tuple, wavelength: float, grid: tuple, plotting_kwargs: dict=dict()) -> None:
        '''
        Inspection tool to plot the sample, its orientation and the incoming, scattered beam 
        and the chosen hkl.
        '''
        # First deal with the plot specifications
        silent = plotting_kwargs.setdefault('silent', True)
        show_sample_dim_ref = plotting_kwargs.setdefault('show_sample_dim_ref', False)
        show_grid_ki = plotting_kwargs.setdefault('show_grid_ki', False)
        show_grid_kf = plotting_kwargs.setdefault('show_grid_kf', False)

        d1, d2 = self.sample_dimensions
        klength = 1.5*max(d1,d2)
        tth = self.lattice.scattering_angle(hkl, wavelength)
        
        arrow_styles = dict(width=klength/100, head_width=klength/20, length_includes_head=True, zorder=0)
        sample_style = dict(ec='black', fc='gray', alpha=0.3, zorder=-20)

        #
        # Sample      
        #
        psi = self.rotate_to_diffract(hkl, wavelength)
        psi_kf = psi+(np.pi-tth)*self.beamline._omega_sense
        
        # Show sample_dimensions_reference arrows
        if show_sample_dim_ref:
            n1 = 5*self.lattice.hkl2Q(self.sample_dimensions_reference[0])
            n2 = 5*self.lattice.hkl2Q(self.sample_dimensions_reference[1])
            n1_rot = np.dot(ms.Rz(psi), n1)
            n2_rot = np.dot(ms.Rz(psi), n2)
            ax.arrow(x=0,y=0,dx=n1_rot[0],dy=n1_rot[1], color='tab:gray', **arrow_styles)
            ax.arrow(x=0,y=0,dx=n2_rot[0],dy=n2_rot[1], color='tab:gray', **arrow_styles)

        l_ki, l_kf = self.central_paths(hkl, wavelength)
        
        rectangle_shift = np.dot(ms.Rz(psi), [d1/2, d2/2, 0])
        sample = mpl.patches.Rectangle(xy=(-rectangle_shift[0], -rectangle_shift[1]), width=d1, height=d2, angle=np.degrees(psi), **sample_style)
        ax.add_artist(sample)

        Nx, Ny = grid
        sdx, sdy = d1/Nx/2, d2/Ny/2
        it, Ntot = 0, Nx*Ny
        for dx in np.linspace(-d1/2+sdx, d1/2-sdx, Nx):
            for dy in np.linspace(-d2/2+sdy, d2/2-sdy, Ny):
                if not silent:
                    it += 1
                    print(f'Working on the element {it}/{Ntot}')

                l_ki = self.rectangle_element_radius(self.sample_dimensions, (dx, dy), psi)
                l_kf = self.rectangle_element_radius(self.sample_dimensions, (dx, dy), psi_kf)
                d_rot = np.dot(ms.Rz(psi), [dx,dy,0])

                ax.scatter(d_rot[0], d_rot[1], color='tab:red', s=0.5)

                if show_grid_ki:
                    ax.plot([d_rot[0]-l_ki, d_rot[0]], [d_rot[1], d_rot[1]])
                
                if show_grid_kf:
                    l_kf_rot = np.dot(ms.Rz(psi-psi_kf), [l_kf,0,0])
                    ax.plot([d_rot[0]-l_kf_rot[0], d_rot[0]], [d_rot[1]-l_kf_rot[1], d_rot[1]])
        
        # main ki
        ki = np.array([klength,0])
        ax.arrow(x=-klength,y=0,dx=ki[0],dy=ki[1], color='tab:red', **arrow_styles)
        ax.annotate(f'k$_i$', xy=(-klength/2,klength/20), color='tab:red')
        
        # main kf
        kf = klength*np.array([np.cos(tth), self.beamline._omega_sense*np.sin(tth)])
        ax.arrow(x=0,y=0,dx=kf[0],dy=kf[1], color='tab:red', **arrow_styles)
        ax.annotate(f'k$_f$', xy=(kf[0]/2+klength/20, kf[1]/2), color='tab:red')

        # Q
        Q = kf-ki
        ax.arrow(x=0,y=0,dx=Q[0],dy=Q[1], color='tab:purple', **arrow_styles)
        ax.annotate(f'Q$_{{{hkl}}}$', xy=Q, color='tab:purple')
        
        return