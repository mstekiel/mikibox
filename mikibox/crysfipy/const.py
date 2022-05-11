# Copyright 2014-2018 Petr Čermák, Jan Zubáč and Karel Pajskr
# This file is part of CrysFiPy.
# CrysFiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# CrysFiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# <http://www.gnu.org/licenses/>.
 
from math import pi

_h = 6.626075540e-34  # Jsec     Planck constant h
_e = 1.6021773349e-19 # Coulomb  electron charge 
_c = 2.99792458e8   # m/sec      Speed of light
_m = 9.109389754e-31  # Kg       Mass of electron m 
_gn = -3.82608545  # g-factor of the neutron

_hq = _h/2/pi        #                h bar
_gamn = _gn/2/_hq    #gyromagnetic ratio of the neutron
_r0 = _e*_e/_m*1e-7  #classical radius of electron
R0 = _r0*_gn/2  #strenght of dipolar neutron electron interaction

uB_SI = 9.27400968 * 1e-24          # J/T
kB_SI = 1.38064880 * 1e-23          # J/K
NA    = 6.02214129 * 1e+23          # 1 / mol
eV2K    = 11.6
uB = uB_SI/kB_SI                    # uB ... 0.67171

invcm2meV = 100*1000*_h*_c/_e

# conversion from [mol/m3] to intern [T/uB]

C1 = NA * uB_SI * 4*pi * 1e-7        # 1/chi [T/uB] = C1 * 1/chi [mol/m3] C1 = 7.0182e-06

# convertion from [mol/m3] to [mol/emu]

C2 = 4*pi/1e6                       # 1/chi [mol/emu] = C2 * 1/chi [mol/m3] C2 = 1.2566e-05

# conversion from [T/uB] to [mol/emu]

C3 = 10/NA/uB_SI                    # = 1/C1*C2 %              1/chi [mol/emu] = C3 * 1/chi [mol/m3] C3 = 1.7905

class ion:
    """Ion information object"""

    def __init__(self, ionstr):
        self.name = ionstr.lower()
        # M = [J gJ Alpha Beta Gamma]
        M = { 
       # ion    J     gJ
		"ce" : [2.5,  6.0/ 7.0,   -2.0/35.0              ,  2.0/315.0                  ,  0.0                            ],
        "pr" : [4.0,  4.0/ 5.0,   -2.0**2*13/3**2/5**2/11, -2.0**2/3**2/5/11**2        ,  2.0**4*17/3**4/5/7/11**2/13    ],
        "nd" : [4.5,  8.0/11.0,    7.0/1089.0            , -136.0/467181.0             , -1615.0/     42513471.0         ],
        "pm" : [4.0,  3.0/ 5.0,    2.0*7/3/5/11**2       ,  2.0**3*7*17/3**3/5/11**3/13,  2.0**3*17*19/3**3/7/11**2/13**2],
        "sm" : [2.5,  2.0/ 7.0,   13.0/3**2/5/7          ,  2.0*13/3**3/5/7/11         ,  0.0                            ],
        "tb" : [6.0,  3.0/ 2.0,   -1.0/99.0              ,  2.0/        16335.0        ,  1.0/(3**4*7*11**2*13)          ],
        "dy" : [7.5,  4.0/ 3.0,   -2.0/3**2/5/7          , -2.0**3/3**3/5/7/11/13      ,  2.0**2/3**3/7/11**2/13**2      ],
        "ho" : [8.0,  5.0/ 4.0,   -1.0/2/3**2/5**2       , -1.0/2/3/5/7/11/13          , -5.0/3**3/7/11**2/13**3         ],
        "er" : [7.5,  6.0/ 5.0,    4.0/(3**2*5**2*7)     ,  2.0/(3**2*5*7*11*13)       ,  8.0/(3**3*7*11**2*13)          ],
        "tm" : [6.0,  7.0/ 6.0,    1.0/3**2/11           ,  2.0**3/3**4/5/11**2	       , -5.0/3**4/7/11**2/13            ],
        "yb" : [3.5,  8.0/ 7.0,    2.0/3**2/7            , -2.0/3/5/7/11               ,  2.0**2/3**3/7/11/13            ],
        }[self.name]

        self.J     = M[0]
        self.J2p1  = int(2 * M[0] + 1)
        self.gJ    = M[1]
        self.Alpha = M[2]
        self.Beta  = M[3]
        self.Gamma = M[4]

    def __str__(self):
        return "%s3+: J = %d, gJ = %.2f" % (self.name.title(), self.J, self.gJ)


