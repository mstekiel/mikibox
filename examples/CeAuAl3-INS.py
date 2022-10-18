import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer

import sys
sys.path.append('C:/Users/Stekiel/Documents/GitHub/mikibox')
import mikibox as ms


# Analysis of the crystal field effects in CeAuAl3
#
# Based on:
# https://doi.org/10.1088/1361-648X/aac408

calculateTAS = True
calculateTOF = True
calculateFormFactor = False


cefion = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'),[0,0,0], ["t", 1.203, -0.001, 0.244])

lattice = ms.Lattice(4.35,4.35,10.6,90,90,90)
print(cefion.cfp)

print(cefion)

start_time = timer()

if calculateTAS:
    fig, ax = plt.subplots()

    e = np.linspace(-10,40,1000)
    temperature = 10

    for hkl in [0,0,4], [1,1,0]:
        De, Dint = ms.crysfipy.neutronint(cefion,temperature,lattice.hkl2k(hkl), Ei=1.1*np.max(e))
        mainTransitions = (np.where(Dint> 0.1*np.max(Dint)))
        
        spectrum = np.zeros(e.shape)
        for Etr, Itr in zip(De, Dint):
            spectrum += Itr*np.exp(-(Etr-e)**2/0.2)

        label = f'Q=({hkl[0]} {hkl[1]} {hkl[2]})'
        #label = f'T={temperature} K'
        ax.plot(e, spectrum, label=label)

    ax.legend()
    fig.savefig('CeAuAl3-TAS-spectra.png')
    
    
if calculateTOF:
    fig, ax = plt.subplots(nrows=3, figsize=(4.2,12), tight_layout=True)

    e = np.linspace(-10,40,1000)
    temperature = 10
    
    selectedEnergies = [cefion.energies[n] for n in [0,2,4]]
    
    Is = [[] for n in range(len(selectedEnergies))]

    N = 100
    hs = np.linspace(-1,1,N)*5
    ls = np.linspace(-1,1,N)*8
    
    Qs = lattice.hkl2k([[h,0,l] for h in hs for l in ls])

    for Q in Qs:
        De, Dint = ms.crysfipy.neutronint(cefion,temperature,Q, Ei=1.1*np.max(e))
        
        for it,selectedEnergy in enumerate(selectedEnergies):
            selectedTransitions = np.where(np.abs(De-selectedEnergy)<1e-5)
            Is[it].append(np.sum(Dint[selectedTransitions]))
               
    
    for it in range(len(selectedEnergies)):
        I = np.reshape(Is[it],(N,N)).T
        X, Y = np.meshgrid(hs,ls)
        
        ax[it].set_xlabel('H')
        ax[it].set_ylabel('L')

        ax[it].set_title(f'Cut h=h, k=0, l=l, E={selectedEnergies[it]:.3f} meV')
        mesh = ax[it].pcolormesh(X, Y, I, shading='auto', cmap=plt.get_cmap('jet')) #PiYG
        
        fig.colorbar(mesh, ax=ax[it])

    #ax.legend()
    fig.savefig('CeAuAl3-TOF-spectra.png')

if calculateFormFactor:
    fig, ax = plt.subplots()

    Q = np.linspace(0,15,1000)
    ff = cefion.ion.mff(Q)
    
    it = 400
    print(Q[it], ff[it])
    ax.plot(Q, ff)


    #ax.legend()
    fig.savefig('CeAuAl3-form-factor.png')
    
end_time = timer()
print(f'Total time = {end_time-start_time} s')