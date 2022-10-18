import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

from timeit import default_timer as timer

import sys
sys.path.append('C:/Users/Stekiel/Documents/GitHub/mikibox')
import mikibox as ms


# Analysis of the crystal field effects in CePdAl3
lattice = ms.Lattice(6.91,6.97,10.6,90,90,90)

calculateTTAS = False
calculateTAS = True
calculateTOF = False


# Following calculation is to cross check the calculations for the orthorhombic case with Ce
# Source: Klicpera et al, PHYSICAL REVIEW B 95, 085107 (2017)
#cefion_CePd2Ga2 = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'),[0,0,0], ["o", 0.33, 0.472, -0.009, 0.111, 0.055])

# Published results:  E1=7.2 meV,   E2=12.2 meV
# Calculated results: E1=7.091 meV, E2=11.772 meV

def fit_CEF_pars(CEFparameters, constraints):
    def residuals(CEFpars,constraints):
        energy_levels = constraints[:6]
        Iratios = constraints[6:]
        try:
            cefion_CePdAl3 = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'),[0,0,0], ['o', *CEFpars])

            # Matching to the measured energy
            energy_resd = cefion_CePdAl3.energies-energy_levels
            
            # Matching the measured intensity
            Irat = []
            for hkl in [[2,2,0],[0,0,4]]:
                De, Dint = ms.crysfipy.neutronint(cefion_CePdAl3,10, Q=lattice.hkl2k(hkl), Ei=25, scheme='single-crystal')
                E1 = cefion_CePdAl3.energies[2] # should be the 2.08 meV transition
                E2 = cefion_CePdAl3.energies[4] # should be the 7.15 meV transition
                I = [0,0]

                for it,selectedEnergy in enumerate([E1,E2]):
                    selectedTransitions = np.where(np.abs(De-selectedEnergy)<1e-5)
                    I[it] += np.sum(Dint[selectedTransitions])
                    
                Irat.append(I[1]/I[0])
                
            intensity_resd = 1e1*(np.array(Irat)-Iratios)
            
            # Total residuum
            residuum = np.concatenate((energy_resd,intensity_resd))
        except ValueError:
            # Complex energies obtained with current set of Bij
            # To reject that solution high residdum is given
            residuum = 1e5*np.ones(len(constraints))
            
        return residuum
        
    from scipy.optimize import leastsq      
    popt, pcov = leastsq(func=residuals, x0=CEFparameters, args=constraints)
    
    score = np.sum(np.power(residuals(popt, constraints),2))
    
    return popt, score

 



#initial_Bij = [0.21601157, 0.15105534, 0.00812939, 0.02302813, 0.09675795]
#initial_Bij = [ 0.00939777,  0.39526846,  0.0012433,   0.02353135,  0.10997198]
#initial_Bij = [-0.11443176,0.32133869, -0.01027173, -0.01867925,0.09454572]
initial_Bij = [1.98770910e-01, 1.51547490e-01, 3.82500000e-03, 3.70048800e-02, 1.01858000e-01]
#initial_Bij = [0.91557101, 0.52525836, 0.15848653, 0.34204767, 0.12747969]


fit_Bij, score = fit_CEF_pars(initial_Bij, constraints=[0,0,2.08,2.08,7.15,7.15,2,0.8])

cefion_CePdAl3 = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'),[0,0,0], ['o', *fit_Bij])

print('Bfit:',fit_Bij, score)
print(cefion_CePdAl3.cfp)
print(cefion_CePdAl3)










# Other calculations
if calculateTTAS:
    fig, ax = plt.subplots(tight_layout=True)
    
    ax.set_xlim(-12,12)
    ax.set_ylim(0,0.06)
    ax.set_xlabel('Energy (meV)')
    ax.set_ylabel('S(Q,E)')

    e = np.linspace(-12,12,1000)

    for temperature in [2,10,25,50,75,100,150,200]:
        
        label = f'T={temperature:3d} K'
        
        for hkl in [[2,2,0]]:
            Q = lattice.hkl2k(hkl)
            De, Dint = ms.crysfipy.neutronint(cefion_CePdAl3,temperature, Q=Q, Ei=1.1*np.max(e), scheme='single-crystal')
           
            
            mainTransitions = (np.where(Dint> 0.01*np.max(Dint)))
            
            sigma = np.exp(temperature/110)
            scaling = 0.04/0.9
            spectrum = np.zeros(e.shape)
            for Etr, Itr in zip(De, Dint):
                # Simulate instrumental convolution
                spectrum += scaling*Itr*np.exp(-(Etr-e)**2/sigma/2)/np.sqrt(sigma*2*np.pi)
            
        spectrum += 1e4*np.exp(-(e)**2/0.1)


        ax.plot(e, spectrum, label=label)

    ax.legend(ncol=2, loc='upper center', shadow=True)
    fig.savefig('CePdAl3-TTAS-spectra.png', dpi=400)
    
if calculateTAS:
    fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
        
    e = np.linspace(-5,10,1000)
    temperature = 0.1
    
    ax.set_xlim(-1,10)
    ax.set_xlabel('Energy (meV)')
    ax.set_ylabel('S(Q,E)')

    for hkl in [2,2,0], [0,0,4], 'powder':
        
        if isinstance(hkl,str):
            label = f'powder average'
            Q = lattice.hkl2k([2,2,0])
            De, Dint = ms.crysfipy.neutronint(cefion_CePdAl3,temperature, Q=ms.norm(Q), Ei=1.1*np.max(e), scheme='powder')
        else:
            label = f'Q=({hkl[0]} {hkl[2]} {hkl[1]})'
            Q = lattice.hkl2k(hkl)
            De, Dint = ms.crysfipy.neutronint(cefion_CePdAl3,temperature, Q=Q, Ei=1.1*np.max(e), scheme='single-crystal')
       
        
        mainTransitions = (np.where(Dint> 0.01*np.max(Dint)))
        
        spectrum = np.zeros(e.shape)
        for Etr, Itr in zip(De, Dint):
            # Simulate instrumental convolution
            spectrum += Itr*np.exp(-(Etr-e)**2/0.1)


        ax.plot(e, spectrum, label=label)

    ax.legend()
    fig.savefig('CePdAl3-TAS-spectra.png', dpi=400)
    
if calculateTOF:
    fig, ax = plt.subplots(nrows=3, figsize=(4.2,12), tight_layout=True)

    e = np.linspace(-10,40,1000)
    temperature = 10
    
    selectedEnergies = [cefion_CePdAl3.energies[n] for n in [0,2,4]]
    
    Is = [[] for n in range(len(selectedEnergies))]

    N = 200
    ind1 = np.linspace(-1,1,N)*3
    ind2 = np.linspace(-1,1,N)*7
    
    Qs = lattice.hkl2k([[h,h,k] for h in ind1 for k in ind2])

    for Q in Qs:
        De, Dint = ms.crysfipy.neutronint(cefion_CePdAl3,temperature,Q, Ei=15, scheme='single-crystal')
        
        for it,selectedEnergy in enumerate(selectedEnergies):
            selectedTransitions = np.where(np.abs(De-selectedEnergy)<1e-5)
            Is[it].append(np.sum(Dint[selectedTransitions]))
                
    
    for it in range(len(selectedEnergies)):
        I = np.reshape(Is[it],(N,N))
        X, Y = np.meshgrid(ind2,ind1)
        
        ax[it].set_xlabel('(0 k 0)')
        ax[it].set_ylabel('(h 0 h)')

        ax[it].set_title(f'Cut (hkh), E={selectedEnergies[it]:.3f} meV')
        mesh = ax[it].pcolormesh(X, Y, I, shading='auto', cmap=plt.get_cmap('jet'), vmin=0.5, vmax=2.2) #PiYG
        
        fig.colorbar(mesh, ax=ax[it])
        
        ax[it].xaxis.set_major_locator(ticker.FixedLocator([-6,-4,-2,0,2,4,6]))

    #ax.legend()
    fig.savefig('CePdAl3-TOF-spectra.png', dpi=400)