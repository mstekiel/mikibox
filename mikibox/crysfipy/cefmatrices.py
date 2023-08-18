'''
Additions by Michal Stekiel

The matrices and conventions need to be cross-checked against some reliable book.
Most important things so far:
 1. The eigenvector convention is that the first entry corresponds to highest spin, i.e.
    for the J=3/2 system |3/2> = [1,0,0,0].
 2. The CEF operators correspond to the Stevens notation.
'''

from numpy import diag, linspace, conj, transpose, sqrt, eye, dot
from numpy.linalg import matrix_power as mp

# J matrices
def J_z(J):
    return diag(linspace(J,-J,int(2*J+1)))

def J_y(J):
	return -.5j * (J_plus(J) - J_minus(J))

def J_x(J):
	return .5 * (J_plus(J) + J_minus(J))



def J_plus(J):
	p1 = linspace(-J,J-1,int(2*J))
	p1 = sqrt(J*(J+1) - p1*(p1+1))
	return diag(p1,1)

def J_minus(J):
	return J_plus(J).conj().transpose()



# Stevens operators
# Cross checked with the McPhase manual
# https://www2.cpfs.mpg.de/~rotter/homepage_mcphase/manual/node132.html

def O_20(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jz = J_z(J)
	E = eye(J2p1)

	return 3*Jz*Jz - JJ * E

def O_22(J):
	Jplus = J_plus(J)
	Jminus = J_minus(J)

	return 0.5 * (mp(Jplus, 2) + mp(Jminus, 2))
    
def O_2m2(J):
	Jplus = J_plus(J)
	Jminus = J_minus(J)

	return -0.5j * (mp(Jplus, 2) - mp(Jminus, 2))

def O_40(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jz = J_z(J)
	E = eye(J2p1)

	return 35 * mp(Jz, 4) + (25 - 30 * JJ)*mp(Jz, 2) + E * JJ * (3 * JJ - 6)


def O_42(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jz = J_z(J)
	Jplus = J_plus(J)
	Jminus = J_minus(J)
	E = eye(J2p1)

    # helper matrices:	
	M_1 = 7 * mp(Jz, 2) - E*(JJ + 5)
	M_2 = mp(Jplus, 2) + mp(Jminus, 2)

	return 0.25 * (dot(M_1, M_2) + dot(M_2, M_1))

def O_4m2(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jz = J_z(J)
	Jplus = J_plus(J)
	Jminus = J_minus(J)
	E = eye(J2p1)

    # helper matrices:	
	M_1 = 7 * mp(Jz, 2) - E*(JJ + 5)
	M_2 = mp(Jplus, 2) - mp(Jminus, 2)

	return -0.25j * (dot(M_1, M_2) + dot(M_2, M_1))
    
def O_43(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jz = J_z(J)
	Jplus = J_plus(J)
	Jminus = J_minus(J)

    # helper matrices:	
	M_1 = Jz
	M_2 = mp(Jplus, 3) + mp(Jminus, 3)
	
	return 0.25 * (dot(M_1, M_2) + dot(M_2, M_1))
    
def O_4m3(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jz = J_z(J)
	Jplus = J_plus(J)
	Jminus = J_minus(J)

    # helper matrices:	
	M_1 = Jz
	M_2 = mp(Jplus, 3) - mp(Jminus, 3)
	
	return -0.25j * (dot(M_1, M_2) + dot(M_2, M_1))


def O_44(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jplus = J_plus(J)
	Jminus = J_minus(J)
	
	return 0.5 * (mp(Jplus, 4) + mp(Jminus, 4))
    
def O_4m4(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jplus = J_plus(J)
	Jminus = J_minus(J)
	
	return -0.5j * (mp(Jplus, 4) - mp(Jminus, 4))


def O_60(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jz = J_z(J)
	E = eye(J2p1)

	return 231 * mp(Jz, 6) + mp(Jz, 4) * (735 - 315 * JJ) + mp(Jz, 2)*(105*JJ**2 - 525*JJ + 294) + E * (-5*JJ**3 + 40 * JJ**2 - 60*JJ)

def O_62(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)
	
	E = eye(J2p1)
	Jz = J_z(J)
	Jplus = J_plus(J)
	Jminus = J_minus(J)
	
    # helper matrices:	
	M_1 = 33 * mp(Jz, 4) - mp(Jz, 2) * (18 * JJ + 123) + E * (JJ**2 + 10*JJ + 102)
	M_2 = mp(Jplus, 2) + mp(Jminus, 2)

	return 0.25 * (dot(M_1, M_2) + dot(M_2, M_1))

def O_6m2(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)
	
	E = eye(J2p1)
	Jz = J_z(J)
	Jplus = J_plus(J)
	Jminus = J_minus(J)
	
    # helper matrices:	
	M_1 = 33 * mp(Jz, 4) - mp(Jz, 2) * (18 * JJ + 123) + E * (JJ**2 + 10*JJ + 102)
	M_2 = mp(Jplus, 2) - mp(Jminus, 2)

	return -0.25j * (dot(M_1, M_2) + dot(M_2, M_1))
    
def O_63(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jz = J_z(J)
	Jplus = J_plus(J)
	Jminus = J_minus(J)

    # helper matrices:	
	M_1 = 11 * mp(Jz, 3) - Jz * (59 + 3*JJ)
	M_2 = mp(Jplus, 3) + mp(Jminus, 3)
	
	return 0.25 * (dot(M_1, M_2) + dot(M_2, M_1))

def O_6m3(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jz = J_z(J)
	Jplus = J_plus(J)
	Jminus = J_minus(J)

    # helper matrices:	
	M_1 = 11 * mp(Jz, 3) - Jz * (59 + 3*JJ)
	M_2 = mp(Jplus, 3) - mp(Jminus, 3)
	
	return -0.25j * (dot(M_1, M_2) + dot(M_2, M_1))
    
def O_64(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	E = eye(J2p1)
	Jz = J_z(J)
	Jplus = J_plus(J)
	Jminus = J_minus(J)

    # helper matrices:	
	M_1 = 11 * mp(Jz, 2) - E * (JJ + 38)
	M_2 = mp(Jplus, 4) + mp(Jminus, 4)

	return 0.25 * (dot(M_1, M_2) + dot(M_2, M_1))

def O_6m4(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	E = eye(J2p1)
	Jz = J_z(J)
	Jplus = J_plus(J)
	Jminus = J_minus(J)

    # helper matrices:	
	M_1 = 11 * mp(Jz, 2) - E * (JJ + 38)
	M_2 = mp(Jplus, 4) - mp(Jminus, 4)

	return -0.25j * (dot(M_1, M_2) + dot(M_2, M_1))

def O_66(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jplus = J_plus(J)
	Jminus = J_minus(J)
	
	return 0.5 * (mp(Jplus, 6) + mp(Jminus, 6))

def O_6m6(J):
	JJ = J*(J+1)
	J2p1 = int(2*J + 1)

	Jplus = J_plus(J)
	Jminus = J_minus(J)
	
	return -0.5j * (mp(Jplus, 6) - mp(Jminus, 6))