import numpy as np
import h5py


def find_dataset(h5_file): 
    '''Automatically find the dataset name in the h5 file containing the input movie.

    Inputs: 
        h5_file (h5py.File): the input h5 file containing the movie.

    Outputs:
        dset (str): the dataset name in the h5 file containing the input movie.
    '''
    list_dset = list(h5_file.keys())
    num_dset = len(list_dset)
    if num_dset == 1:
        dset = list_dset[0]
    else:
        list_size = np.zeros(num_dset, dtype='uint64')
        for (ind,dset) in enumerate(list_dset):
            if isinstance(h5_file[dset], h5py.Dataset):
                list_size[ind] = h5_file[dset].size
        ind_dset = list_size.argmax()
        if list_size[ind_dset] > 0:
            dset = list_dset[ind_dset]
        else:
            raise(KeyError('Cannot find any dataset. Please do not put the movie under any group'))
    return dset

