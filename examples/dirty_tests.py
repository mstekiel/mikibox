'''
Some quick look-throughs to examine new components of the library etc
'''

import mikibox as ms
import mikibox.crysfipy as cfp
import mikibox.crystallography as mcryst
import mikibox.instruments as LSF

import numpy as np
import matplotlib.pyplot as plt

ZEBRA = LSF.PSI_ZEBRA()
ErB2_lattice = ms.crystallography.Lattice([3.269, 3.269, 3.78, 90,90,120])

ErB2_absorption_corection = ms.crystallography.AbsorptionCorrection(\
    lattice=ErB2_lattice, beamline=ZEBRA, absorption_coefficient=0.01, \
    sample_shape='rectangle', sample_dimensions=(1, 2.5), sample_dimensions_reference=((-1,2,0),(0,0,1)))

ac = ErB2_absorption_corection
hkl_list = [(-1,2,0),(-1,2,1),(-1,2,2),(-1,2,3),(0,0,1)]

print(hkl_list)

print('Attenuation following the `central_path` method')
print( ErB2_absorption_corection.attenuation(hkl_list, 1.178, method='central') )

print('Attenuation following the `grid_elements` method')
print( ErB2_absorption_corection.attenuation(hkl_list, 1.178, method='grid', method_kwargs=dict(Nx=2, Ny=2)) )

# psi = np.linspace(0.01,2*np.pi-0.1, 101)
# radius1 = np.array([ac.rectangle_radius((1, 2.5), p) for p in psi])
# radius2 = np.array([ac.rectangle_element_radius((1, 2.5), (0.1, 0.5), p) for p in psi])
# radius3 = np.array([ac.rectangle_element_radius2((1, 2.5), (0.2, 0.6), p) for p in psi])

#print(psi)
#print()
#print(np.arctan2(np.sin(psi), np.cos(psi)))

if True:
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)

    # ax.plot(radius1*np.cos(psi), radius1*np.sin(psi), color='tab:blue', label='central_path')
    # ax.plot(radius2*np.cos(psi), radius2*np.sin(psi), color='tab:green', label='grid_path')

    # ax.plot(radius3*np.cos(psi)-0.2, radius3*np.sin(psi)-0.6, color='tab:red', label='grid_path_poly')

    #ax.scatter(psi, np.arctan2(np.sin(psi), np.cos(psi)))

    plotting_kwargs = dict(silent=True, show_sample_dim_ref=False, show_grid_ki=False, show_grid_kf=True)

    ac.plot_beam_paths_grid(ax, (-1,2,0), 1.178, (5,7), plotting_kwargs=plotting_kwargs)

    ac.rotate_to_diffract

    ax.legend()
    
    
    fig.savefig('test.png')
    plt.close()