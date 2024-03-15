import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

from timeit import default_timer as timer

import sys
sys.path.append('C:/Users/Stekiel/Documents/GitHub/mikibox')
import mikibox as ms


# Analysis of the crystal field effects in CePdAl3
lattice = ms.crystallography.Lattice([3.275,3.275,3.784,90,90,120])

calculateMagnetism = True

Bij = [1.26987, -0.00062433, -2.86435e-07, 1.01125e-05]
# Bij = [1.26987, 0, 0, 0]

CEFpars = ms.crysfipy.CEFpars('D6h', Bij, 'meV')

cefion_ErB2 = ms.crysfipy.CEFion(ms.crysfipy.Ion('Er'),[0,0,0], CEFpars)
cefion = cefion_ErB2
print(CEFpars)
print(cefion_ErB2.cfp)
print(cefion_ErB2)








if calculateMagnetism:
    temperature=0.1
    temperatures = np.linspace(1,300,601)

    fig, axs = plt.subplots(figsize=(5,10), nrows=3, tight_layout=True)


    # Set up the susceptibility calculated from magnetization
    ax = axs[0]

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Magnetic susceptibility ($\mu_B^2$/ion)')

    chi_100 = ms.crysfipy.susceptibility(cefion, temperatures , (1,0,0), method='magnetization')
    chi_001 = ms.crysfipy.susceptibility(cefion, temperatures , (0,0,1), method='magnetization')

    ax.plot(temperatures, chi_100, marker='^', color='red', label='$\chi$ || [100]', mfc=None)
    ax.plot(temperatures, chi_001, marker='o', color='blue', label='$\chi$ || [001]', mfc=None)

    ax.set_yscale('log')
    ax.legend(title=f'ErB$_2$ magnetization')

    # Set up the susceptibility calculated from perturbation
    ax = axs[1]

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Magnetic susceptibility ($\mu_B^2$/ion)')
    #ax.set_ylim(0,4)

    chi_100 = ms.crysfipy.susceptibility(cefion, temperatures , (1,0,0), method='perturbation')
    chi_010 = ms.crysfipy.susceptibility(cefion, temperatures , (0,1,0), method='perturbation')
    chi_001 = ms.crysfipy.susceptibility(cefion, temperatures , (0,0,1), method='perturbation')

#    print(chi_100)

    ax.plot(temperatures, chi_100, marker='^', color='red', label='$\chi$ || [100]', mfc=None)
    ax.plot(temperatures, chi_010, marker='s', color='green', label='$\chi$ || [010]', mfc=None)
    ax.plot(temperatures, chi_001, marker='o', color='blue', label='$\chi$ || [001]', mfc=None)

    ax.set_yscale('log')
    ax.legend(title=f'ErB$_2$ perturbation')

    # Set up the magnetization
    ax = axs[2]

    ax.set_xlabel('Applied field (T)')
    ax.set_ylabel('Magnetic moment ($\mu_B$/ion)')

    fields = np.linspace(1e-5, 1000, 101)
    Ma, Mb, Mc = [], [], []
    for field in fields:
        Ma.append(ms.crysfipy.magnetization(cefion, temperature, (field,0,0))[0])
        Mb.append(ms.crysfipy.magnetization(cefion, temperature, (0,field,0))[1])
        Mc.append(ms.crysfipy.magnetization(cefion, temperature, (0,0,field))[2])



    ax.plot(fields, Ma, marker='^', color='red', label='H || [100]')
    ax.plot(fields, Mb, marker='s', color='green', label='H || [010]')
    ax.plot(fields, Mc, marker='o', color='blue',label='H || [001]')

    ax.legend(title=f'ErB$_2$, T={temperature} K')

    fig.savefig(r'C:\Users\Stekiel\Documents\GitHub\mikibox\examples\ErB2-magnetism-mikibox.png',dpi=200)