import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

from timeit import default_timer as timer

import sys
sys.path.append('C:/Users/Stekiel/Documents/GitHub/mikibox')
import mikibox as ms

K2meV = 0.086173



# Following calculation is to cross check the calculations for the orthorhombic case with Ce
# Source: Klicpera et al, PHYSICAL REVIEW B 95, 085107 (2017)
#cefion_CePd2Ga2 = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'),[0,0,0], ["o", 0.33, 0.472, -0.009, 0.111, 0.055])



CEFpars = ms.crysfipy.CEFpars('D4', [-1.0, 0.001, 0.025, 5e-6, 5e-4], 'K')
cefion = ms.crysfipy.CEFion(ms.crysfipy.Ion('Dy'),[0,0,0], CEFpars)

print(cefion.cfp)
print(cefion)

temperature=0.1
temperatures = [10] #np.linspace(0.1,300,601)

fig, axs = plt.subplots(figsize=(5,10), nrows=3, tight_layout=True)


# Set up the susceptibility calculated from magnetization
ax = axs[0]

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Magnetic susceptibility ($\mu_B^2$/Ce)')

chi_100 = ms.crysfipy.susceptibility(cefion, temperatures , [1,0,0], method='magnetization')
chi_001 = ms.crysfipy.susceptibility(cefion, temperatures , [0,0,1], method='magnetization')

ax.plot(temperatures, chi_100, marker='^', color='red', label='$\chi$ || [100]', mfc=None)
ax.plot(temperatures, chi_001, marker='o', color='blue', label='$\chi$ || [001]', mfc=None)

ax.legend(title=f'DyRhIn$_5$ magnetization')

# Set up the susceptibility calculated from perturbation
ax = axs[1]

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Magnetic susceptibility ($\mu_B^2$/Ce)')
#ax.set_ylim(0,4)

chi_100 = ms.crysfipy.susceptibility(cefion, temperatures , [1,0,0], method='perturbation')
chi_001 = ms.crysfipy.susceptibility(cefion, temperatures , [0,0,1], method='perturbation')

print(chi_100)

ax.plot(temperatures, chi_100, marker='^', color='red', label='$\chi$ || [100]', mfc=None)
ax.plot(temperatures, chi_001, marker='o', color='blue', label='$\chi$ || [001]', mfc=None)

ax.legend(title=f'DyRhIn$_5$ perturbation')

# Set up the magnetization
ax = axs[2]

ax.set_xlabel('Applied field (T)')
ax.set_ylabel('Magnetic moment ($\mu_B$/Ce)')

fields = np.linspace(1e-5, 50, 101)
Ma, Mc = [], []
for field in fields:
    Ma.append(ms.crysfipy.magnetization(cefion, temperature, [field,0,0])[0])
    Mc.append(ms.crysfipy.magnetization(cefion, temperature, [0,0,field])[2])



ax.plot(fields, Ma, marker='^', color='red', label='H || [100]')
ax.plot(fields, Mc, marker='o', color='blue',label='H || [001]')

ax.legend(title=f'DyRhIn$_5$, T={temperature} K')

fig.savefig('DyRhIn5-mikibox.png',dpi=200)
