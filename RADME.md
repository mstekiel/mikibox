import numpy as np

import sys
sys.path.append('C:/Users/Michal/Documents/GitHub/michal')
import michal as ms

v1 = [-1,1,1]
v2 = [0,0,1]
angle = np.radians(180)

#print(ms.rotate(angle*np.array(v1)/ms.norm(v1)))
#print(np.degrees(ms.angle(v1,v2)))
print(ms.perppart(v1,v2))