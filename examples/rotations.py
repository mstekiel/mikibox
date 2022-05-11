import numpy as np
import matplotlib.pyplot as plt
import yaml

import sys
sys.path.append('C:/Users/Michal/Documents/GitHub/mikibox')
import mikibox as ms



# Look into the force constants
def calculateForces(fc,rij):
    '''
    Calculate force and force components based on the second rank force constant tensor
    fc: matrix representing the force tensor between two atoms [3x3 float]
    rij:  vector connecting two atoms [3x1 float]
    '''
     
    # Diagonalize th force constant matrix
    alpha = ms.angle([1,0,0],rij)
    _n = np.cross([1,0,0],rij)
    n = _n*alpha/ms.norm(_n)
    T = ms.rotate(n)
    T_inv = ms.rotate(-n)

    fc_Ryz = np.matmul(T, np.matmul(fc,T_inv) )

    kL = fc_Ryz[0,0]
    kT = ms.norm([fc_Ryz[1,1],fc_Ryz[2,2]])

    return kL, kT
    
    
fc = np.array([[1,1,0],[-1,1,0],[0,0,0.4]])
rij = np.array([1,1,0])

#print(calculateForces(fc,rij))

print(dir(ms))
print(ms.rotate([1,0,0]))