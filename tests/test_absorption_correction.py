import unittest

import numpy as np

from mikibox.crystallography import AbsorptionCorrection, Lattice
from mikibox.instruments import Beamline

class AbsorptionCorrectionTest(unittest.TestCase):
    def setUp(self):
        # Set up the test case with the tolerances for vectors tests
        self.rtol = 1e-7
        self.atol = 1e-15

        # Set up dummy absorption correction problem to draw the internal functions from
        self.absorptionCorrection = AbsorptionCorrection(Lattice([1,1,1,90,90,90]), Beamline, \
            absorption_coefficient=1, sample_shape='circle', sample_dimensions=1, sample_dimensions_reference=(1,0,0))

    def test_rectangle_radius(self):
        ac = self.absorptionCorrection
   
        np.testing.assert_allclose( ac.rectangle_radius((2, 3), 0), 1, \
                                    atol=self.atol, rtol=self.rtol)

        np.testing.assert_allclose( ac.rectangle_radius((2, 3), np.pi/2), 1.5, \
                                    atol=self.atol, rtol=self.rtol)

        np.testing.assert_allclose( ac.rectangle_radius((2, 2), np.pi/4), np.sqrt(2), \
                                    atol=self.atol, rtol=self.rtol)

        np.testing.assert_allclose( ac.rectangle_radius((2*np.sqrt(3), 2), np.pi/6), 2, \
                                    atol=self.atol, rtol=self.rtol)

if __name__ == '__main__':
    unittest.main()