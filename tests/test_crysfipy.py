import unittest
import numpy as np
import mikibox as ms
import mikibox.crysfipy as cfp
import mikibox.crysfipy.constants as C

class CrysFiPyInternalFunctionsTest(unittest.TestCase):
    # Set up the test case with the tolerances for vectors tests
    def setUp(self):
        self.rtol = 1e-7
        self.atol = 1e-15


    def test_BoltzmannFactors(self):
        # Check if the boltzmann distribution works well for the degenerated cases.

        p1 = cfp.boltzman_population([0,0], 10)
        np.testing.assert_allclose( [0.5, 0.5], p1, \
                            atol=self.atol, rtol=self.rtol)



class CrysFiPyTest(unittest.TestCase):
    # Set up the test case with the tolerances for vectors tests
    def setUp(self):
        self.rtol = 1e-7
        self.atol = 1e-15
        
    def assert_eigenvectors(self, nominal_ev, possible_evs, rtol, atol):
        '''
        Helper function to assess the validity of the degenerated eigenvectors.
        Only one eigenvector from the candidates list should match the nominal one.
        '''
        
        ev_fails = 0
        msg = ''
        
        # Take a look into all candidates and save the exceptions' messages
        for candidate_ev in possible_evs:
            try:
                np.testing.assert_allclose( nominal_ev, candidate_ev, \
                                            rtol=rtol, atol=atol )      
            except AssertionError as am:
                ev_fails += 1
                msg += str(am)
               
        # Only one eigenvector should match
        if ev_fails==len(possible_evs)-1:
            pass
        elif ev_fails==len(possible_evs):
            print(f'Eigenvector assertion failed: None of the possible eigenvector matches.')
            print(msg)
            raise AssertionError
        elif ev_fails<len(possible_evs)-1:
            print(f'Eigenvector assertion failed: More than one of possible eigenvector matches.')
            print(msg)
            raise AssertionError
        
    def test_CeAuAl3(self):
        # Test the CeAuAl3 case based on:
        # Cermak et al., PNAS, April 2, 2019, vol. 116, no. 14, 6695–6700
        # Magnetoelastic hybrid excitations in CeAuAl3
        # www.pnas.org/cgi/doi/10.1073/pnas.1819664116
        ce = cfp.CEFion(cfp.Ion("Ce"), [0,0,0], cfp.CEFpars('C4h', [1.203, -0.001, 0.244], 'meV'))

        # Compare energies
        # All levels are two-fold degenerated, with excitation energies 2.96 meV and 24.27 meV
        np.testing.assert_allclose(ce.energies, [ 0.,0.,4.96,4.96,24.27,24.27], rtol=1e-2, atol=1e-2)
        
        # Compare wavefunctions
        # ground state:     |+-1/2>
        # first excited:   -alpha |-+3/2> + beta  |+-5/2>
        # second excited:   beta  |+-3/2> + alpha |-+5/2>
        # with alpha=0.931, beta=0.364
        alpha, beta = 0.931, 0.364
        rtol, atol = 1e-2, 1e-3
        
        # It's all bit tricky because the eigenvectors of degenerated states can be in any order. Also, the eigenvector is physically the same even if all coefficients are multiplied by any phase factor exp(i*phi) eg. -1
        # The eigenvector goes from |5/2> to |-5/2>
        
        # Ground state
        allowed_evs = [[0,0,1,0,0,0], [0,0,0,1,0,0]]
        self.assert_eigenvectors(ce.eigenvectors[:,0], allowed_evs, self.rtol, self.atol)
        self.assert_eigenvectors(ce.eigenvectors[:,1], allowed_evs, self.rtol, self.atol)
        
        # First excited state, requires multiplying by -1 phase factor
        allowed_evs = [[-beta,0,0,0,alpha,0], [0,alpha,0,0,0,-beta]]
        self.assert_eigenvectors(ce.eigenvectors[:,2], allowed_evs, rtol, atol)
        self.assert_eigenvectors(ce.eigenvectors[:,3], allowed_evs, rtol, atol)
        
        # Second excited state
        allowed_evs = [[0,beta,0,0,0,alpha], [alpha,0,0,0,beta,0]]
        self.assert_eigenvectors(ce.eigenvectors[:,4], allowed_evs, rtol, atol)
        self.assert_eigenvectors(ce.eigenvectors[:,5], allowed_evs, rtol, atol)
        
    def test_Tb3inY3Al5O12(self):
        # Heavy lifting
        # Test case based on:
        # Gruber et al. PRB 69, 115103 (2004)
        
        # TODO
        # Doesnt work
        
        # The final set of parameters involves
        cefpars = np.array([461,165,-169,-1720,-900,-1324,-621,599,-561])*C.invcm2meV
        tb = cfp.CEFion(cfp.Ion("Tb"), [0,0,0], cfp.CEFpars('C2v', cefpars, 'meV'))
        
        print(tb.hamiltonian)

                                    
if __name__ == '__main__':
    unittest.main()