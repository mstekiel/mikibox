import numpy as np

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