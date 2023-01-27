import unittest
import numpy as np
import mikibox as ms

class VectorsTest(unittest.TestCase):
    # Set up the test case with the tolerances for vectors tests
    def setUp(self):
        self.rtol = 1e-7
        self.atol = 1e-15
        
    def test_norm(self):
        np.testing.assert_allclose( ms.norm([1,0,0]), 1, \
                                    atol=self.atol, rtol=self.rtol)
                                    
        np.testing.assert_allclose( ms.norm([-3,0,3]), 3*np.sqrt(2), \
                                    atol=self.atol, rtol=self.rtol)
                                    
        np.testing.assert_allclose( ms.norm([1,1,1]), np.sqrt(3), \
                                    atol=self.atol, rtol=self.rtol)
                                    
        np.testing.assert_allclose( ms.norm([1,0,0,1,1]), np.sqrt(3), \
                                    atol=self.atol, rtol=self.rtol)
    
    def test_angle(self):
        np.testing.assert_allclose( ms.angle([1,0,0],[1,0,0]), 0, \
                                    atol=self.atol, rtol=self.rtol)
                                    
        np.testing.assert_allclose( ms.angle([1,0,0],[1,1,0]), np.radians(45), \
                                    atol=self.atol, rtol=self.rtol)                                    
        np.testing.assert_allclose( ms.angle([1,1,0],[1,1,1]), np.arctan2(1,np.sqrt(2)), \
                                    atol=self.atol, rtol=self.rtol)

        # Go around and check special values
        for phi in np.linspace(0,180,25):
            phiR = np.radians(phi)
            calculated_angle = np.degrees(ms.angle([1,0,0],[np.cos(phiR),np.sin(phiR),0]))
            np.testing.assert_allclose(calculated_angle , phi, \
                                    atol=self.atol, rtol=self.rtol)

        # All perpendicular
        for phi in np.linspace(0,180,25):
            phiR = np.radians(phi)
            calculated_angle = np.degrees(ms.angle([0,0,1],[np.cos(phiR),np.sin(phiR),0]))
            np.testing.assert_allclose(calculated_angle , 90, \
                                    atol=self.atol, rtol=self.rtol) 

    def test_perp_matrix(self):
        # Q = [0,0,1] -> projects on the xy plane
        Q = np.array([0,0,1])
        Qperp = np.array([  [1,0,0],
                            [0,1,0],
                            [0,0,0]])
        np.testing.assert_allclose( ms.perp_matrix(Q), Qperp, \
                                    atol=self.atol, rtol=self.rtol)

    def test_perp_part(self):
        # Perpendicular vectors first
        m = [1,0,0]
        q = [0,0,1]
        r = m
        np.testing.assert_allclose( ms.perp_part(m,q), r, \
                                    atol=self.atol, rtol=self.rtol)

        m = [-12,4,0]
        q = [0,0,1]
        r = m
        np.testing.assert_allclose( ms.perp_part(m,q), r, \
                                    atol=self.atol, rtol=self.rtol)

        m = [1,1,0]
        q = [1,0,0]
        r = [0,1,0]
        np.testing.assert_allclose( ms.perp_part(m,q), r, \
                                    atol=self.atol, rtol=self.rtol)

    def test_cartesian2spherical(self):
        # Basic xyz vectors
        vector = np.array([0,0,1])
        r, theta, phi = 1,0,0
        np.testing.assert_allclose( ms.cartesian2spherical(vector), [r,theta,phi], \
                                    atol=self.atol, rtol=self.rtol)

        vector = np.array([0,1,0])
        r, theta, phi = 1,np.radians(90),np.radians(90)
        np.testing.assert_allclose( ms.cartesian2spherical(vector), [r,theta,phi], \
                                    atol=self.atol, rtol=self.rtol)

        vector = np.array([1,0,0])
        r, theta, phi = 1,np.radians(90),np.radians(0)
        np.testing.assert_allclose( ms.cartesian2spherical(vector), [r,theta,phi], \
                                    atol=self.atol, rtol=self.rtol)
                                    
        # Some other interesting vectors
        vector = np.array([1,1,0])
        r, theta, phi = np.sqrt(2),np.radians(90),np.radians(45)
        np.testing.assert_allclose( ms.cartesian2spherical(vector), [r,theta,phi], \
                                    atol=self.atol, rtol=self.rtol)
 
        vector = np.array([0,-1,1])
        r, theta, phi = np.sqrt(2),np.radians(45),np.radians(-90)
        np.testing.assert_allclose( ms.cartesian2spherical(vector), [r,theta,phi], \
                                    atol=self.atol, rtol=self.rtol)
                                    
        vector = np.array([1,0,-1])
        r, theta, phi = np.sqrt(2),np.radians(135),np.radians(0)
        np.testing.assert_allclose( ms.cartesian2spherical(vector), [r,theta,phi], \
                                    atol=self.atol, rtol=self.rtol)
                                    
class RotationsTest(unittest.TestCase):
    # Set up the test case with the tolerances for matrix tests
    def setUp(self):
        self.rtol = 1e-7
        self.atol = 1e-15
        
    def test_Rx(self):
        # Test the Rx rotation matrix
        
        # Rotation by zero is an identity matrix
        np.testing.assert_allclose(ms.Rx(0), np.eye(3), atol=self.atol, rtol=self.rtol)
        
        # Right hand rotation by 90 deg changes axes
        R = np.array([  [1,0,0],
                        [0,0,-1],
                        [0,1,0]])
        np.testing.assert_allclose(ms.Rx(np.radians(90)), R, atol=self.atol, rtol=self.rtol)
        
        # Right hand rotation by 180 deg reverts axes
        R = np.array([  [1,0,0],
                        [0,-1,0],
                        [0,0,-1]])
        np.testing.assert_allclose(ms.Rx(np.radians(180)), R, atol=self.atol, rtol=self.rtol)
        
    def test_Ry(self):
        # Test the Ry rotation matrix
        
        # Rotation by zero is an identity matrix
        np.testing.assert_allclose(ms.Ry(0), np.eye(3), atol=self.atol, rtol=self.rtol)
        
        # Right hand rotation by 90 deg changes axes
        R = np.array([  [0,0,1],
                        [0,1,0],
                        [-1,0,0]])
        np.testing.assert_allclose(ms.Ry(np.radians(90)), R, atol=self.atol, rtol=self.rtol)
        
        # Right hand rotation by 180 deg reverts axes
        R = np.array([  [-1,0,0],
                        [0,1,0],
                        [0,0,-1]])
        np.testing.assert_allclose(ms.Ry(np.radians(180)), R, atol=self.atol, rtol=self.rtol)
    
    def test_Rz(self):
        # Test the Rz rotation matrix
        
        # Rotation by zero is an identity matrix
        np.testing.assert_allclose(ms.Rz(0), np.eye(3), atol=self.atol, rtol=self.rtol)
        
        # Right hand rotation by 90 deg changes axes
        R = np.array([  [0,-1,0],
                        [1,0,0],
                        [0,0,1]])
        np.testing.assert_allclose(ms.Rz(np.radians(90)), R, atol=self.atol, rtol=self.rtol)
        
        # Right hand rotation by 180 deg reverts axes
        R = np.array([  [-1,0,0],
                        [0,-1,0],
                        [0,0,1]])
        np.testing.assert_allclose(ms.Rz(np.radians(180)), R, atol=self.atol, rtol=self.rtol)
        
    def test_rotate(self):
        # Test the general rotation matrix
        
        # Rotation by zero around any vector is an identity matrix
        vector, angle = np.array([1,0,0]), 0
        np.testing.assert_allclose( ms.rotate(vector,np.radians(angle)), np.eye(3), \
                                    atol=self.atol, rtol=self.rtol)
                                    
        vector, angle = np.array([1,1,0]), 0
        np.testing.assert_allclose( ms.rotate(vector,np.radians(angle)), np.eye(3), \
                                    atol=self.atol, rtol=self.rtol)
        
        # Compare with the basic xyz rotations
        vector = np.array([1,0,0])
        
        angle = np.radians(90)
        np.testing.assert_allclose( ms.rotate(vector,angle), ms.Rx(angle), \
                                    atol=self.atol, rtol=self.rtol)
                                    
        angle = np.radians(33)
        np.testing.assert_allclose( ms.rotate(vector,angle), ms.Rx(angle), \
                                    atol=self.atol, rtol=self.rtol)
        
        vector = np.array([0,1,0])
        
        angle = np.radians(90)
        np.testing.assert_allclose( ms.rotate(vector,angle), ms.Ry(angle), \
                                    atol=self.atol, rtol=self.rtol)
                                    
        angle = np.radians(55)
        np.testing.assert_allclose( ms.rotate(vector,angle), ms.Ry(angle), \
                                    atol=self.atol, rtol=self.rtol)


        vector = np.array([0,0,1])
        
        angle = np.radians(90)
        np.testing.assert_allclose( ms.rotate(vector,angle), ms.Rz(angle), \
                                    atol=self.atol, rtol=self.rtol)
                                    
        angle = np.radians(77)
        np.testing.assert_allclose( ms.rotate(vector,angle), ms.Rz(angle), \
                                    atol=self.atol, rtol=self.rtol)   


        # Rotation around cube body diagonal permutes the axes
        vector, angle = np.array([1,1,1]), np.radians(120)
        R = np.array([  [0,0,1],
                        [1,0,0],
                        [0,1,0]])
        np.testing.assert_allclose( ms.rotate(vector,angle), R, \
                                    atol=self.atol, rtol=self.rtol)
                                
        vector, angle = np.array([1,1,1]), np.radians(240)
        R = np.array([  [0,1,0],
                        [0,0,1],
                        [1,0,0]])
        np.testing.assert_allclose( ms.rotate(vector,angle), R, \
                                    atol=self.atol, rtol=self.rtol)
                                    
if __name__ == '__main__':
    unittest.main()