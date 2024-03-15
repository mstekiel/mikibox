import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer

#import sys
#sys.path.append('C:/Users/Stekiel/Documents/GitHub/mikibox')
import mikibox as ms

hex = ms.crystallography.Lattice([3.275, 3.275, 3.784, 90, 90, 120])
atoms = [
    ['Er1', (0,0,0), (0,0,1)],
    ['B', (0,0,1), (0,0,0)],
    ['B', (1,0,0), (0,0,0)]
]

sw_erb = ms.spinwaves.Structure(lattice=hex, atoms=atoms, couplings=[], modulation=(0,0))

sw_erb.plot_structure()