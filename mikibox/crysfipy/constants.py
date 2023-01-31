from math import pi

numerical_zero = 1e-12  # python3.9.15 precision for floats seems to be around 1e-15, so lets say everything below 

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
meV2K    = 11.60452
uB = 1e3*uB_SI/_e                    # uB in meV/T

invcm2meV = 100*1000*_h*_c/_e
K2meV = 0.086173

# conversion from [mol/m3] to intern [T/uB]

C1 = NA * uB_SI * 4*pi * 1e-7        # 1/chi [T/uB] = C1 * 1/chi [mol/m3] C1 = 7.0182e-06

# convertion from [mol/m3] to [mol/emu]

C2 = 4*pi/1e6                       # 1/chi [mol/emu] = C2 * 1/chi [mol/m3] C2 = 1.2566e-05

# conversion from [T/uB] to [mol/emu]

C3 = 10/NA/uB_SI                    # = 1/C1*C2 %              1/chi [mol/emu] = C3 * 1/chi [mol/m3] C3 = 1.7905
