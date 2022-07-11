import numpy as np

# Fitting functions and other
def gauss_bkg(x,x0,A,sigma,bkg):
    '''
    Gaussian with constant background.
    
    :math:`f(x) = A exp(-(x-x_0)^2/(2 \\sigma^2)) + bkg`
    
    To convert to intensity of the peak :math:`I = \\sqrt{2 \\pi} A \\sigma`
    '''
    return A*np.exp(-(x-x0)**2/(2*sigma**2)) + bkg
    
def lorentz_bkg(x,x0,A,gamma,bkg):
    '''
    Gaussian with constant background.
    
    :math:`f(x) = \\frac{A}{(1+(x-x_0)^2/\\gamma^2))} + bkg`
    
    To convert to intensity of the peak :math:`I = \\pi A \\gamma`
    '''
    return A/(1+np.power((x-x0)/gamma,2)) + bkg
    
def pseudoVoigt_bkg(x,x0,I,f,eta,bkg):
    '''
    Pseudo-Voigt function.
    '''
    
    return eta*I*gauss_bkg(x,x0,1/(np.sqrt(2*np.pi)*f),f,0) + (1-eta)*I*lorentz_bkg(x,x0,1/(np.pi*f),f,0) + bkg

def gauss_satellites_bkg(x,x0,xs,As,sigmas,bkg):
    '''
    Gaussian with constant background.
    
    :math:`f(x) = A exp(-(x-x_0)^2/(2 \\sigma^2)) + bkg`
    
    To convert to intensity of the peak :math:`I = \\sqrt{2 \\pi} A \\sigma`
    '''
    return As*np.exp(-(x-x0-xs)**2/(2*sigmas**2)) + As*np.exp(-(x-x0+xs)**2/(2*sigmas**2)) + bkg

# Rotations
# All of them are right-handed
def rotate(n, angle):
    '''
    Return a matrix representing the rotation around vector {n} by {angle} radians.
    Length of the {n} vector does not matter.
    '''
    alpha, theta, phi = cartesian2spherical(n)

    return np.matmul(Rz(phi), np.matmul(Ry(theta), np.matmul(Rz(angle), np.matmul(Ry(-theta), Rz(-phi) ))))
    
def Rx(alpha):
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
    
def Ry(alpha):
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])

def Rz(alpha):
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])


# Vectors
def cartesian2spherical(xyz):
    '''
    Return the spherical [r, theta, phi] coordinates of the cartesian vector [x,y,z]
    r > 0
    theta in (0 : pi)
    phi in (-pi : pi)
    '''
    xy = xyz[0]**2 + xyz[1]**2
    r = norm(xyz)
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    phi = np.arctan2(xyz[1], xyz[0])
    return [r,theta,phi]
    
def norm(x):
    return np.sqrt(np.dot(x,x))
    
    
def angle(v1,v2):
    '''
    Return the angle between two vectors
    '''
    
    return np.arccos(np.dot(v1,v2)/norm(v1)/norm(v2))

def perp_matrix(q):
    '''
    Return the matrix representing projection on the plane perpendicular to the given vector q
    '''
    
    # For the sake of speed the matrix is given explicitly based on calculations on paper
    r, theta, phi = cartesian2spherical(q)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    
    return np.array([   [1-st*st*cp*cp, -st*st*sp*cp,   -ct*st*cp],
                        [-st*st*sp*cp,  1-st*st*sp*sp,  -ct*st*sp],
                        [-st*ct*sp,     -st*ct*sp,      1-ct*ct]])
    
def perp_part(m,q):
    '''
    Return the part of vector m that is perpendicular to the vector q
    '''
    # eq = np.array(q)/norm(q)
    # return np.cross(np.cross(eq,m), eq)
    
    return np.dot(perp_matrix(q), m)