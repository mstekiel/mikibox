import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import multiprocessing, os

from timeit import default_timer as timer

import sys
sys.path.append('C:/Users/Stekiel/Documents/GitHub/mikibox')
import mikibox as ms


lattice = ms.crystallography.Lattice([4,4,10.6,90,90,90])

calculateTAS = True
calculateMagnetism = True

num_simulations = 100000

# Following calculation is to cross check the calculations for the orthorhombic case with Ce
# Source: Klicpera et al, PHYSICAL REVIEW B 95, 085107 (2017)
#cefion_CePd2Ga2 = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'),[0,0,0], ["o", 0.33, 0.472, -0.009, 0.111, 0.055])

# Published results:  E1=7.2 meV,   E2=12.2 meV
# Calculated results: E1=7.091 meV, E2=11.772 meV

CEFpars = ms.crysfipy.CEFpars('D4', [-10.26, -0.056, 2.67], 'K')
cefion = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'),[0,0,0], CEFpars, diagonalize=False)
cefion.diagonalize(shiftToZero=False)

print(cefion.cfp)
print(cefion)


if calculateTAS:
    fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
        
    e = np.linspace(-2,25,1000)
    temperature = 2
    
    #ax.set_xlim(-1,10)
    ax.set_xlabel('Energy (meV)')
    ax.set_ylabel('S(Q,E)')

    for hkl in [1.25,1.25,0], [0,0,4]:
        
        label = f'Q=({hkl[0]} {hkl[1]} {hkl[2]})'
        Q = lattice.hkl2Q(hkl)
        De, Dint = ms.crysfipy.neutronint(cefion,temperature, Q=Q, Ei=1.1*np.max(e), scheme='single-crystal')
       
        
        mainTransitions = (np.where(Dint> 0.01*np.max(Dint)))
        
        sigma=0.1
        spectrum = np.zeros(e.shape)
        for Etr, Itr in zip(De, Dint):
            # Simulate instrumental convolution
            spectrum += Itr*np.exp(-(Etr-e)**2/sigma/2)/np.sqrt(sigma*2*np.pi)


        ax.plot(e, spectrum, label=label)

    ax.legend()
    fig.savefig('CeCu2Ge2-TAS-mikibox.png',dpi=200)
    
if calculateMagnetism:
    temperature=0.1
    temperatures = np.linspace(0.1,300,601)

    fig, axs = plt.subplots(figsize=(5,10), nrows=3, tight_layout=True)


    # Set up the susceptibility calculated from magnetization
    ax = axs[0]

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Magnetic susceptibility ($\mu_B^2$/Ce)')

    chi_100 = ms.crysfipy.susceptibility(cefion, temperatures , [1,0,0], method='magnetization')
    chi_001 = ms.crysfipy.susceptibility(cefion, temperatures , [0,0,1], method='magnetization')

    ax.plot(temperatures, chi_100, marker='^', color='red', label='$\chi$ || [100]', mfc=None)
    ax.plot(temperatures, chi_001, marker='o', color='blue', label='$\chi$ || [001]', mfc=None)

    ax.legend(title=f'CeCu$_2$Ge$_2$ magnetization')

    # Set up the susceptibility calculated from perturbation
    ax = axs[1]

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Magnetic susceptibility ($\mu_B^2$/Ce)')
    #ax.set_ylim(0,4)

    chi_100 = ms.crysfipy.susceptibility(cefion, temperatures , [1,0,0], method='perturbation')
    chi_001 = ms.crysfipy.susceptibility(cefion, temperatures , [0,0,1], method='perturbation')

#    print(chi_100)

    ax.plot(temperatures, chi_100, marker='^', color='red', label='$\chi$ || [100]', mfc=None)
    ax.plot(temperatures, chi_001, marker='o', color='blue', label='$\chi$ || [001]', mfc=None)

    ax.legend(title=f'CeCu$_2$Ge$_2$ perturbation')

    # Set up the magnetization
    ax = axs[2]

    ax.set_xlabel('Applied field (T)')
    ax.set_ylabel('Magnetic moment ($\mu_B$/Ce)')

    fields = np.linspace(1e-5, 5, 101)
    Ma, Mc = [], []
    for field in fields:
        Ma.append(ms.crysfipy.magnetization(cefion, temperature, [field,0,0])[0])
        Mc.append(ms.crysfipy.magnetization(cefion, temperature, [0,0,field])[2])



    ax.plot(fields, Ma, marker='^', color='red', label='H || [100]')
    ax.plot(fields, Mc, marker='o', color='blue',label='H || [001]')

    ax.legend(title=f'CeCu$_2$Ge$_2$, T={temperature} K')

    fig.savefig('CeCu2Ge2-magnetism-mikibox.png',dpi=200)
