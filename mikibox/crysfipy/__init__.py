__version__ = '1.1'

# Import main classes
from .ion import Ion
from .cefpars import CEFpars
from .cefion import CEFion

# Import the functions
from .cefmatrices import *
from .observables import *

# Original copyright:

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

# I have done major reworking of the package to make it usable for my own purposes.
# Michal Stekiel