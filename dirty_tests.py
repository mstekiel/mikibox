'''
Some quick look-throughs to examine new components of the library etc
'''

import mikibox as ms
import mikibox.crysfipy as cfp
import mikibox.crysfipy.const as C
import numpy as np


cef_list = ['t', 0.77e-1, 4.4e-4, 0.24e-5, 2.2e-3, 0.34e-5]
tb = cfp.re("Tb", [0,0,0], cef_list)
tb.getlevels()
print()
print(tb.energy)
print()
