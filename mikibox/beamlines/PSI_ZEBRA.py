import numpy as np

def get_refl(DATA,hkl):
    '''
    Get the datablock corresponding to the reflection
    '''
    h,k,l= hkl
    
    refl_id=-1
    for it,DATAblock in enumerate(DATA):
        if DATAblock['h']==h and DATAblock['k']==k and DATAblock['l']==l:
            refl_id = it
            
    if refl_id==-1:
        raise Warning(f'Reflection {hkl} not found in the dataset')
        return None
        
    return DATA[refl_id]

def load(filename, extension=None):
    '''
    Wrapper for loading one of the two filetypes used at ZEBRA
    '''

    # Figure out the format from file extension
    if extension == None:
        if filename[-3:] == 'ccl':
            extension = 'ccl'
        elif filename[-3:] == 'dat':
            extension = 'dat'
        else:
            raise ValueError('Unknown extension of the file')

            
    if extension == 'ccl':
        DATA, HEADER = load_ccl(filename)
    elif extension == 'dat':
        DATA, HEADER = load_dat(filename)
    else:
        raise ValueError('Unknown extension of the file')   
        
    return DATA, HEADER
    
def load_ccl(filename):
    '''
    Read the ccl file containing multiple scans of various reflections
    
    Some additional entries regarding the scan are included in the header.
    
    Returns:
        HEADER : dictionary
            Contains the main header of the ccl file
        DATA : array of dictionaries
            Each dictionary contains a measured reflection. Main data is stored in DATA[n]['counts'] and supplementary DATA[n]['x'], 
            where x is omega, or omega-twotheta for high Q.
    '''
    DATA = []
    HEADER = {}
    
    with open(filename) as ff:
        lines = ff.readlines()

    data_start = -1     
        
    for it,line in enumerate(lines):
        line_records = line.split()
        if len(line_records)>1 and line_records[1]=='=':
            it = line.find('=')
            
            try:
                value = float(line[it+2:])
            except ValueError:
                value = line[it+2:]
                
            HEADER[line_records[0]] = value
            
        if line=='#data\n':
            data_start = it
            
    # Read Npoints for a reflection. It should be the same for all reflections in a single ccl file.
    block_start = data_start+2
    Npoints = int(lines[data_start+3].split()[0])
    block_lines = int(Npoints//10+1+2)
    
    while block_start < len(lines):
        # Read block header
        keys1 = ['Nblock', 'h', 'k', 'l', 'tth', 'om', 'chi', 'phi', '_', '_']
        vals1 = [float(x) for x in lines[block_start].split()]
        keys2  =['Npoints', 'step', 'monitor', '_', '_', 'date', 'time', 'scantype', 'step2']
        vals2 = [x if it in [5,6,7] else float(x) for it,x in enumerate(lines[block_start+1].split())]
        
        # Assign block header
        DATA_subset = {}
        for key,val in zip(keys1,vals1):
            DATA_subset[key] = val
            
        for key,val in zip(keys2,vals2):
            DATA_subset[key] = val
            
        # Read counts
        counts_start = block_start+2
        counts_end = counts_start + block_lines - 2
        counts = np.array([int(cts) for cts in ' '.join(lines[counts_start:counts_end]).split()])
        
        # Asign counts and additional entries
        DATA_subset['counts'] = counts
        x_start = DATA_subset['om'] - Npoints/2*DATA_subset['step']
        #x_end = DATA_subset['om'] + Npoints/2*DATA_subset['step']
        DATA_subset['x'] = x_start+np.arange(0,Npoints,1)*DATA_subset['step']
            
        
        DATA.append(DATA_subset)
        
        block_start += block_lines
    
    return HEADER, DATA

            
def load_dat(filename):
    '''
    Read the dat file containing a single scan.
    
    Some additional entries regarding the scan are included in the header.
    '''
    DATA = []
    HEADER = {}
    
    with open(filename) as ff:
        lines = ff.readlines()
        
    data_start = -1   
    data_end = -1    
        
    for it,line in enumerate(lines):
        line_records = line.split()
        if len(line_records)>1 and line_records[1]=='=':
            it = line.find('=')
            
            try:
                value = float(line[it+2:])
            except ValueError:
                value = line[it+2:]
                
            HEADER[line_records[0]] = value
            
        if line=='#data\n':
            data_start = it
            
        if line=='END-OF-DATA\n':
            data_end = it
            
    data_points = int(lines[data_start+2].split()[0])
    HEADER['data_points'] = data_points
    
    data_header = lines[data_start+3].split()
    HEADER['data_header'] = data_header
    
    # This does not work for qscans so lets omit it
    # scan_line = lines[data_start+1]
    # trimmed_line = scan_line.replace('Scanning Variables:','').replace('Steps:','')
    # scan_variables, scan_steps = trimmed_line.split(',')
    # HEADER['scan_variables'] = scan_variables.replace(' ','')
    # HEADER['scan_steps'] = [float(x) for x in scan_steps.split()]
    
    
    DATA_str = [x.split() for x in lines[data_start+4:data_end]]
    DATA = np.array(DATA_str, dtype=float)

            
    return DATA, HEADER

def load_d23(filenme):
    # Read omega scan from native D23 raw format
    
    DATA = []
    HEADER = OrderedDict()
    
    with open(filename) as ff:
        lines = ff.readlines()

    # Read the numor that resides before the main header
    numor = int(lines[1].split()[0])
    HEADER['number'] = numor

    # Read in the header values, pray that they are always in the same lines
    for it in np.arange(22,31+1):
        linechunks1 = [lines[it][i:i+16] for i in range(0,len(lines[it]),16)]
        keys = [str(n).strip() for n in linechunks1]
        vals = [float(n) for n in lines[it+10].split()]

        for key, val in zip(keys,vals):
            HEADER[key] = val

    # Load the data. Number of columns depends on the type of scans, but can be deduced from number of k-points measured and total number of entries
    nentries = int(lines[45].split()[0])
    nkmes = int(lines[16].split()[5])
    
    ncols = int(nentries/nkmes)
    
    for line in lines[46:]:
        DATA = np.append(DATA, [float(n) for n in line.split()])

    try:
        DATA=DATA.reshape((int(nentries/ncols),ncols))
    except BaseException:
        raise ValueError('FILE:',filename,'  Couldnt reshape DATA array to shape given in the header')

    return DATA, HEADER
    