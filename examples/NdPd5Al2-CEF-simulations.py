import numpy as np
import matplotlib.pyplot as plt
import yaml

import sys
sys.path.append('C:/Users/Michal/Documents/GitHub/mikibox')
import mikibox as ms


# Analysis of the crystal field effects in NdPd5Al2
#
# Based on:
# https://doi.org/10.1088/1361-648X/aac408

calculateINS = False
calculateM = False
calculateSusceptibility = False
calculateHeatCapacity = True

# Create single ion of Nd in NdPd5Al2
c = 0.0000862   # Conversion factor between mK and meV
nd = ms.crysfipy.CEFion(ms.crysfipy.Ion('Nd'),[0,0,0], ["t", -1006*c, 5.23*c,-91.0*c, -0.802*c, -9.65*c])

# Quick check for SmRhIn5
c = 0.0862   # Conversion factor between K and meV
nd = ms.crysfipy.CEFion(ms.crysfipy.Ion('Tm'),[0,0,0], ["t", -1.2*c, -0.9*c,1*c, 0, 0])

print(nd.cfp)

print(nd)

if calculateINS:
    colors = {'2':'black', '10':'blue', '20':'red', '40':'green', '80':'orange'}

    fig, ax = plt.subplots()

    e = np.linspace(-10,20,1000)
    spectrum = np.zeros(e.shape)

    #for Q in [0,0,4], [1,1,0], [1,0,0]:
    Q = 'powder'
    
    ax.set_ylim(0,10)
    
    for T in 2,10,20,40,80:
        De, Dint = ms.crysfipy.neutronint(nd,T,Q)
        for Etr, Itr in zip(De, Dint):
            spectrum += Itr*np.exp(-(Etr-e)**2/0.2)

        label = f'Q=({Q[0]} {Q[1]} {Q[2]})'
        label = f'T={T} K'
        ax.plot(e, spectrum, label=label, color=colors[str(T)])

    ax.legend()
    fig.savefig('CePdAl3-INS-spectra.png')
    

if calculateM:
    fig, ax = plt.subplots()
   
#    ax.set_ylim(0,10)
    
    H = np.linspace(-5,5,50)
    M = []    
    for h in H:
        M.append(ms.crysfipy.magnetization(nd, 20, [0,h,0]))

    Mx, My, Mz = np.transpose(M)
    ax.plot(H, Mx, label='Mx', color='tab:red')
    ax.plot(H, My, label='My', color='tab:green')
    ax.plot(H, Mz, label='Mz', color='tab:blue')

    ax.legend()
    fig.savefig('CePdAl3-magnetization.png')
    
if calculateSusceptibility:
    fig, ax = plt.subplots()
   
#    ax.set_ylim(0,10)
    
    T = np.linspace(2,300,100)
    susc_a = ms.crysfipy.susceptibility(nd, T, [1,0,0], method='magnetization')
    susc_c = ms.crysfipy.susceptibility(nd, T, [0,0,1], method='magnetization')

    ax.scatter(T, 1/susc_a, label='chi_a', color='blue')
    #ax.scatter(T, 1/susc_c, label='chi_c', color='black')


    ax.legend()
    fig.savefig('CePdAl3-susceptibility.png')
    
if calculateHeatCapacity:
    fig, ax = plt.subplots()
   
#    ax.set_ylim(0,10)
    
    T = np.linspace(1e-2,30,100)
    Z, E, S, Cv = ms.crysfipy.heatCapacity(nd, T)
    #susc_c = ms.crysfipy.susceptibility(nd, T)
    
    #ax.scatter(T, Z, label='Z', color='blue')
    ax.scatter(T, E, label='E', color='black')
    ax.plot(T, S, label='S', color='red')
    #ax.plot(T, Cv, label='Cv', color='green')


    ax.legend()
    fig.savefig('CePdAl3-heatCapacity.png')