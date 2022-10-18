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



CEFpars = ms.crysfipy.CEFpars('D4', [-16, 0.55, 0.61], 'K')
cefion = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'),[0,0,0], CEFpars)

print(cefion.cfp)
print(cefion)

temperature=65
fields = np.linspace(1e-5, 50, 101)
Ma, Mc = [], []
for field in fields:
    Ma.append(ms.crysfipy.magnetization(cefion, temperature, [field,0,0])[0])
    Mc.append(ms.crysfipy.magnetization(cefion, temperature, [0,0,field])[2])


fig, ax = plt.subplots()

ax.set_xlabel('Applied field (T)')
ax.set_ylabel('Magnetic moment ($\mu_B$/Ce)')

ax.plot(fields, Ma, marker='^', color='red', label='H || [100]')
ax.plot(fields, Mc, marker='o', color='blue',label='H || [001]')

ax.legend(title=f'CeRhIn$_5$, T={temperature} K')
fig.savefig('CeRhIn5-mikibox.png',dpi=200)
