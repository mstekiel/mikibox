import numpy as np

from typing import List

class Ion:
    r"""
    Class representing an isolated rare-earth ion and its fundamental parameters.
    
    Attributes:
        J : int/2
            Quantum number describing the total angular momentum L-S
        J2p1 : int 
            Degeneracy of the manifold
        gJ  : float
            Lande coefficient
        Alpha : float
            :math:`\alpha` parameter
        Beta : float
            :math:`\beta` parameter
        Gamma : float
            :math:`\gamma` parameter
        mffCoefficients : dict
            Dictionary of coefficients, used to calculate the magnetic form factor
    """

    def __init__(self, ion_name: str):
        self.name = ion_name.lower()

        atomicDatabase = {
        "ce": {"J":2.5, "gJ":0.8571428571428571,
                "alpha":-5.7142857143e-02,      "beta":6.3492063492e-03,        "gamma":0.0000000000e+00,
                "js":{  "j0":[0.2953, 17.6846, 0.2923, 6.7329, 0.4313, 5.3827, -0.0194],
                        "j2":[0.9809, 18.063, 1.8413, 7.7688, 0.9905, 2.8452, 0.012] }},
        "pr": {"J":4.0, "gJ":0.8,
                "alpha":-2.1010101010e-02,      "beta":-7.3461891644e-04,       "gamma":6.0994000388e-05,
                "js":{  "j0":[0.0504, 24.9989, 0.2572, 12.0377, 0.7142, 5.0039, -0.0219],
                        "j2":[0.8734, 18.9876, 1.5594, 6.0872, 0.8142, 2.415, 0.0111] }},
        "nd": {"J":4.5, "gJ":0.7272727272727273,
                "alpha":6.4279155188e-03,       "beta":-2.9110772912e-04,       "gamma":-3.7987959158e-05,
                "js":{  "j0":[0.054, 25.0293, 0.3101, 12.102, 0.6575, 4.7223, -0.0216],
                        "j2":[0.6751, 18.3421, 1.6272, 7.26, 0.9644, 2.6016, 0.015] }},
        "sm": {"J":2.5, "gJ":0.2857142857142857,
                "alpha":4.1269841270e-02,       "beta":2.5012025012e-03,        "gamma":0.0000000000e+00,
                "js":{  "j0":[0.0288, 25.2068, 0.2973, 11.8311, 0.6954, 4.2117, -0.0213],
                        "j2":[0.4707, 18.4301, 1.4261, 7.0336, 0.9574, 2.4387, 0.0182] }},
        "tb": {"J":6.0, "gJ":1.5,
                "alpha":-1.0101010101e-02,      "beta":1.2243648607e-04,        "gamma":1.1212132424e-06,
                "js":{  "j0":[0.0177, 25.5095, 0.2921, 10.5769, 0.7133, 3.5122, -0.0231],
                        "j2":[0.2892, 18.4973, 1.1678, 6.7972, 0.9437, 2.2573, 0.0232] }},
        "dy": {"J":7.5, "gJ":1.3333333333333333,
                "alpha":-6.3492063492e-03,      "beta":-5.9200059200e-05,       "gamma":1.0349660699e-06,
                "js":{  "j0":[0.1157, 15.0732, 0.327, 6.7991, 0.5821, 3.0202, -0.0249],
                        "j2":[0.2523, 18.5172, 1.0914, 6.7362, 0.9345, 2.2082, 0.025] }},
        "ho": {"J":8.0, "gJ":1.25,
                "alpha":-2.2222222222e-03,      "beta":-3.3300033300e-05,       "gamma":-9.9515968263e-08,
                "js":{  "j0":[0.0566, 18.3176, 0.3365, 7.688, 0.6317, 2.9427, -0.0248],
                        "j2":[0.2188, 18.5157, 1.024, 6.707, 0.9251, 2.1614, 0.0268] }},
        "er": {"J":7.5, "gJ":1.2,
                "alpha":2.5396825397e-03,       "beta":4.4400044400e-05,        "gamma":2.6909117818e-05,
                "js":{  "j0":[0.0586, 17.9802, 0.354, 7.0964, 0.6126, 2.7482, -0.0251],
                        "j2":[0.171, 18.5337, 0.9879, 6.6246, 0.9044, 2.1004, 0.0278] }},
        "tm": {"J":6.0, "gJ":1.1666666666666667,
                "alpha":1.0101010101e-02,       "beta":1.6324864810e-04,        "gamma":-5.6060662121e-06,
                "js":{  "j0":[0.0581, 15.0922, 0.2787, 7.8015, 0.6854, 2.7931, -0.0224],
                        "j2":[0.176, 18.5417, 0.9105, 6.5787, 0.897, 2.0622, 0.0294] }},
        "yb": {"J":3.5, "gJ":1.1428571428571428,
                "alpha":3.1746031746e-02,       "beta":-1.7316017316e-03,       "gamma":1.4800014800e-04,
                "js":{  "j0":[0.0416, 16.0949, 0.2849, 7.8341, 0.6961, 2.6725, -0.0229],
                        "j2":[0.157, 18.5553, 0.8484, 6.5403, 0.888, 2.0367, 0.0318] }},
        }

        elementData = atomicDatabase[self.name]

        self.J     = elementData['J']
        self.J2p1  = int(2 * self.J + 1)
        self.gJ    = elementData['gJ']
        self.Alpha = elementData['alpha']
        self.Beta  = elementData['beta']
        self.Gamma = elementData['gamma']
        self.mffCoefficients = elementData['js']

    def __str__(self):
        return "%s: J = %d, gJ = %.2f" % (self.name.title(), self.J, self.gJ)
        
    def _LandeGFactor(self, S: float,L: int) -> float:
        # Maybe for the future implementation, when the initialization will be done with S and L not with J
        return 1.5 + (S*(S+1) - L*(L+1))/(2*self.J*(self.J+1))
        
    def _m_in_vacuum(self) -> float:
        # Calculate magnetic moment of an isolated ion in the units of Bohr magnetons.
        return self.gJ*np.abs(self.J)

    def _m_dynamic(self) -> float:
        # Calculate magnetic moment of an isolated ion when calculating susceptibility.
        return self.gJ*np.abs(self.J*(self.J+1))
        
    def mff(self,Q: List[float]) -> np.ndarray:
        '''
        Calculate the magnetic form factor at given momentum transfer Q. From dipole approximation (small Q):
        | :math:`f_m(Q) = j_0(Q) + \\frac{2-g_J}{g_J} j_2(Q)`
        
        Functions :math:`j_n` are tabulated internally, :math:`g_J` is the Lande factor.
        
        Parameters:
            Q
                List of the selected Q values. Note, Q is not a vector in reciprocal space, but its magnitude.
        
        Returns:
            Array of calculated values
        '''
        s = np.divide(Q, 4*np.pi)
        coefs0 = self.mffCoefficients['j0']
        coefs2 = self.mffCoefficients['j2']
        
        j0 = coefs0[0]*np.exp(-coefs0[1]*s**2) + coefs0[2]*np.exp(-coefs0[3]*s**2)  + \
                coefs0[4]*np.exp(-coefs0[5]*s**2) +coefs0[6]

        j2 = s**2*(coefs2[0]*np.exp(-coefs2[1]*s**2) + coefs2[2]*np.exp(-coefs2[3]*s**2)+\
               coefs2[4]*np.exp(-coefs2[5]*s**2) + coefs2[6])
        
        return j0 + j2*(2-self.gJ)/self.gJ


def _composeTables(filename: str) -> str:
    '''
    Internal function that makes the tables of elements with principal values required for calculations. It is not well implemented, but it works and makes things easy.
    '''
    
    # Principal factors as implemented originally by Czech guys
    M = { 
       # ion    J     gJ
		"ce" : [2.5,  6.0/ 7.0,   -2.0/35.0              ,  2.0/315.0                  ,  0.0                            ],
        "pr" : [4.0,  4.0/ 5.0,   -2.0**2*13/3**2/5**2/11, -2.0**2/3**2/5/11**2        ,  2.0**4*17/3**4/5/7/11**2/13    ],
        "nd" : [4.5,  8.0/11.0,    7.0/1089.0            , -136.0/467181.0             , -1615.0/     42513471.0         ],
        #"pm" : [4.0,  3.0/ 5.0,    2.0*7/3/5/11**2       ,  2.0**3*7*17/3**3/5/11**3/13,  2.0**3*17*19/3**3/7/11**2/13**2], # Promethium is missing from form factors database
        "sm" : [2.5,  2.0/ 7.0,   13.0/3**2/5/7          ,  2.0*13/3**3/5/7/11         ,  0.0                            ],
        "tb" : [6.0,  3.0/ 2.0,   -1.0/99.0              ,  2.0/        16335.0        ,  1.0/(3**4*7*11**2*13)          ],
        "dy" : [7.5,  4.0/ 3.0,   -2.0/3**2/5/7          , -2.0**3/3**3/5/7/11/13      ,  2.0**2/3**3/7/11**2/13**2      ],
        "ho" : [8.0,  5.0/ 4.0,   -1.0/2/3**2/5**2       , -1.0/2/3/5/7/11/13          , -5.0/3**3/7/11**2/13**3         ],
        "er" : [7.5,  6.0/ 5.0,    4.0/(3**2*5**2*7)     ,  2.0/(3**2*5*7*11*13)       ,  8.0/(3**3*7*11**2*13)          ],
        "tm" : [6.0,  7.0/ 6.0,    1.0/3**2/11           ,  2.0**3/3**4/5/11**2	       , -5.0/3**4/7/11**2/13            ],
        "yb" : [3.5,  8.0/ 7.0,    2.0/3**2/7            , -2.0/3/5/7/11               ,  2.0**2/3**3/7/11/13            ]}
        
    # Form factors database by J.P. Brown
    jDatabase = {}
    with open(filename, 'r') as ff:
        lines = ff.readlines()
        
    for line in lines:
        if line[0] == '#':
            continue
        else:
            fields = line.replace('=',' ').split()
            name = fields[1].lower()
            keys = fields[3::2]
            values = np.array(fields[4::2], dtype=float)
            jorder = keys[0][3]
            
            if name in jDatabase:
                jDatabase[name][f'j{jorder}'] = values
                
            else:
                jDatabase[name] = {f'j{jorder}' : values}
            
    atomicDatabase = {}
    for name in M:
        if name == 'ce':
            oxidation = '2'
        else:
            oxidation = '3'
            
        extendedName = name+oxidation
        
        atomicDatabase[name] = {'J':M[name][0], 'gJ':M[name][1], 'alpha':M[name][2], 'beta':M[name][3], 'gamma':M[name][4], 'js':jDatabase[extendedName]}
     
     
    precision = 10
    out = ['atomicDatabase = {']
    for name in atomicDatabase:
        out.append(f'"{name}": {{"J":{atomicDatabase[name]["J"]}, "gJ":{atomicDatabase[name]["gJ"]}, ')
        out.append(f'\t"alpha":{atomicDatabase[name]["alpha"]:.{precision}e},\t"beta":{atomicDatabase[name]["beta"]:.{precision}e},\t"gamma":{atomicDatabase[name]["gamma"]:.{precision}e},')
        out.append(f'\t"js":{{\t"j0":{list(atomicDatabase[name]["js"]["j0"])},')
        out.append(f'\t\t"j2":{list(atomicDatabase[name]["js"]["j2"])} }}}},')
        
    out.append('}')
        
    return '\n'.join(out)
        
if __name__ == '__main__':
    tables = _composeTables('./atoms-form-factors.txt')
    print(tables)
    
    
    