import numpy as np

replace_dict = {'[':'', ']':'', ',':'', '\'':''}
DATA = []
with open('pandahpc-results2.txt','r') as ff:
    for line in ff:
        fline = line[1::2]
        ttable = fline.maketrans(replace_dict)
        fline = fline.translate(ttable).strip()
        
        records = fline.split()
        
        if len(records)==7:
            DATA.append(records[1:])

DATA = np.array(DATA, dtype=float)

print(DATA[np.argsort(DATA[:,5])])