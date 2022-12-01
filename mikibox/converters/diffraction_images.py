import numpy as np

def load_crysalis_img(filename: str, header_length: int=6576, data_shape: tuple=(775,1215)) -> tuple:
    '''
    Load data from an `img` file created by CrysAlis into an nd.array.

    The header and data lengths can be inferred from the header values. NHEADER nad NX*NY.
    The rest is not deciphered yet.

    Comments:
        The header contains some binary data that is ignored.

        Data shape is not 2^N, but it works. Also, there is something peculiar about the byte format, it looks like it gives 127 to zero counts.

        There is some leftover information at the end of the file, that is ignored.
    '''
    with open(filename,'br') as ff:
        file_header = ff.read(header_length)
        file_data = ff.read(data_shape[0]*data_shape[1])

    # Cast to unsigned int
    data = []
    for b in file_data:
        data.append(int(b))

    return file_header, np.array(data).reshape(data_shape)