import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

from timeit import default_timer as timer

import sys
sys.path.append('C:/Users/Stekiel/Documents/GitHub/mikibox')
import mikibox as ms

K2meV = 0.086173



# Following calculation is to cross check the calculations for the LiHoF4
# Source 1: Wendl et al, Nature, 609, 65-70, (2022)
# Source 2: Chakraborty et al., PRB, 70, 144411, (2004)


# For C4h point group encoded list of parameters is:
# ["B20", "B40", "B44", "B4m4", "B60", "B64", "B6m4"]

CEFpars = ms.crysfipy.CEFpars('C4h', [-0.696, 4.06e-3, 0.0418, 0, 4.64e-6, 8.12e-4, 1.137e-4], 'K')
print(CEFpars)

# Print ground state properties
cefion = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ho'),[0,0,0], CEFpars)
print(cefion)




CEFenergies = []

fields = np.linspace(0,1,26)
for Bx in fields:
    cefion = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ho'),[Bx,0,0], CEFpars, diagonalize=False)
    cefion.diagonalize(shiftToZero=False)
    
    print(Bx, cefion.Jx[0,0],cefion.Jy[0,0],cefion.Jz[0,0])

    CEFenergies.append(cefion.energies/K2meV+244.34380728)
    
CEFenergies = np.transpose(CEFenergies)


# Make plots on two energy scales to compare with published data
fig, (axL, axR) = plt.subplots(figsize=(8,4), ncols=2, tight_layout=True)

axL.set_ylim(-60,550)
axR.set_ylim(-30,30)

for ax in axL, axR:
    ax.set_xlabel('B$_x$ (T)')
    ax.set_ylabel('E (K)')

for excitation in CEFenergies:
    axL.plot(fields, excitation)
    axR.plot(fields, excitation)

fig.savefig('./LiFo4-calculated-scheme.png', dpi=200)

'''
temperature=1.3
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

ax.legend(title=f'NdRhIn$_5$')

# Set up the susceptibility calculated from perturbation
ax = axs[1]

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Magnetic susceptibility ($\mu_B^2$/Ce)')
ax.set_ylim(0,4)

chi_100 = ms.crysfipy.susceptibility(cefion, temperatures , [1,0,0], method='perturbation')
chi_001 = ms.crysfipy.susceptibility(cefion, temperatures , [0,0,1], method='perturbation')

ax.plot(temperatures, chi_100, marker='^', color='red', label='$\chi$ || [100]', mfc=None)
ax.plot(temperatures, chi_001, marker='o', color='blue', label='$\chi$ || [001]', mfc=None)

ax.legend(title=f'NdRhIn$_5$')

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

ax.legend(title=f'NdRhIn$_5$, T={temperature} K')

fig.savefig('NdRhIn5-mikibox.png',dpi=200)
'''