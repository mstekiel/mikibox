import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import multiprocessing, os

from timeit import default_timer as timer

import sys
sys.path.append('C:/Users/Stekiel/Documents/GitHub/mikibox')
import mikibox as ms


# Analysis of the crystal field effects in CePdAl3
lattice = ms.Lattice(6.91,6.97,10.6,90,90,90)

calculateTAS = True
calculateTOF = True
runMonteCarlo = True

num_simulations = 100000

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

 

# Monte Carlo part
def find_Bij(p):
    rng = np.random.default_rng()
    initial_Bij = rng.random((5, 1))
    
    try:
        fit_Bij, score = fit_CEF_pars(initial_Bij, constraints=[0,0,2.08,2.08,7.15,7.15,2,0.8])

        if score<1e1:
            print('Bfit:',fit_Bij,score)        
    except ValueError:
        pass
        
    return score
        

if __name__ == '__main__':
    start_time = timer()
    print(f'Starting MC simulation on {os.cpu_count()} processors.')

    with multiprocessing.Pool(os.cpu_count()) as pool:
        scores = pool.map(find_Bij, range(num_simulations))
        
    end_time = timer()
    print(f'Total time = {end_time-start_time} s')