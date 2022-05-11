import numpy as np
import matplotlib.pyplot as plt
import yaml

import sys
sys.path.append('C:/Users/Michal/Documents/GitHub/mikibox')
import mikibox as ms


# Analysis of the crystal field effects in CePdAl3
#
# The main point is to be able to reconstruct the butterfly maps of the CEF intensity distribution in the reciprocal space

# Create single ion of Ce in CePdAl3
c = 0.0000862
ce = ms.crysfipy.re('Nd',[0,0,0], ["t", -1006*c, 5.23*c,-91.0*c, -0.802*c, -9.65*c])

print(ce)
De, Dint = ms.crysfipy.neutronint(ce,10,[1,0,0])
# print(ms.crysfipy.neutronint(ce,10,[0,1,0]))

colors = {'2':'black', '10':'blue', '20':'red', '40':'green', '80':'orange'}

if True:
    fig, ax = plt.subplots()

    e = np.linspace(-10,20,1000)
    spectrum = np.zeros(e.shape)

    #for Q in [0,0,4], [1,1,0], [1,0,0]:
    Q = 'powder'
    
    ax.set_ylim(0,10)
    
    for T in 2,10,20,40,80:
        De, Dint = ms.crysfipy.neutronint(ce,T,Q)
        for Etr, Itr in zip(De, Dint):
            spectrum += Itr*np.exp(-(Etr-e)**2/0.4)

        label = f'Q=({Q[0]} {Q[1]} {Q[2]})'
        label = f'T={T} K'
        ax.plot(e, spectrum, label=label, color=colors[str(T)])

    ax.legend()
    fig.savefig('CePdAl3-spectra.png')