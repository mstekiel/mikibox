import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer

import mikibox as ms

# Following calculation is to cross check the calculations for the orthorhombic case with Ce
# Source: Klicpera et al, PHYSICAL REVIEW B 95, 085107 (2017)
#cefion_CePd2Ga2 = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'),[0,0,0], ["o", 0.33, 0.472, -0.009, 0.111, 0.055])



CEFpars = ms.crysfipy.CEFpars('D4', [-16, 0.55, 0.61], 'K')
cefion = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'), (0,0,0), CEFpars)

print(cefion.cfp)
print(cefion)

temperature=65
fields = np.linspace(1e-5, 50, 101)
Ma, Mb, Mc = [], [], []
for field in fields:
    Ma.append(ms.crysfipy.magnetization(cefion, temperature, (field,0,0))[0])
    Mb.append(ms.crysfipy.magnetization(cefion, temperature, (0,field,0))[1])
    Mc.append(ms.crysfipy.magnetization(cefion, temperature, (0,0,field))[2])


fig, ax = plt.subplots()

ax.set_xlabel('Applied field (T)')
ax.set_ylabel('Magnetic moment ($\mu_B$/Ce)')

ax.plot(fields, Ma, marker='s', color='red', label='H || [100]')
ax.plot(fields, Mb, marker='^', color='green', label='H || [010]')
ax.plot(fields, Mc, marker='o', color='blue',label='H || [001]')

ax.legend(title=f'CeRhIn$_5$, T={temperature} K')
fig.savefig('./examples/CeRhIn5-mikibox.png',dpi=200)
