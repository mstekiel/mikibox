# Crystallography subpackage of the mikibox package
#
# Contains various classes representing fundamental crystallographic concepts
# as well as some helperfunctions to analyze the data.

__version__ = '1.0'

# NEXT VERSION: update after the AbsorptionCorrection problem is implemented with central and grid methods

# Import main classes
from .lattice import Lattice
from .absorption_correction import AbsorptionCorrection

# Import the functions
from .crystallography_functions import *