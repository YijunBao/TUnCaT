import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
import h5py
from utils import find_dataset

if (sys.version_info.major+sys.version_info.minor/10)>=3.8:
    from multiprocessing.shared_memory import SharedMemory
    from traces_from_masks_mp_shm_neighbors import traces_bgtraces_from_masks_shm_neighbors
else:
    from traces_from_masks_mp_mmap_fn_neighbors import traces_bgtraces_from_masks_mmap_neighbors
from traces_from_masks_numba_neighbors import traces_bgtraces_from_masks_numba_neighbors
from use_nmfunmix_mp_MSE_novideo import use_nmfunmix


def run_TUnCaT(Exp_ID, filename_video, filename_masks, dir_traces, list_alpha=[0], Qclip=0, \
        th_pertmin=1, epsilon=0, th_residual=False, nbin=1, \
        bin_option='downsample', multi_alpha=True, flexible_alpha=True):
    ''' Unmix the traces of all neurons in a video, and obtain the unmixed traces and the mixing matrix. 
        The video is stored in "filename_video", and the neuron masks are stored in "filename_masks".
        The output traces will be stored in "dir_traces".
    Inputs: 
        Exp_ID (str): The name of the video.
        filename_video (str): The file path (including file name) of the video.
            The video file must be a ".h5" file. 
        filename_masks (str): The file path (including file name) of the neuron masks.
            The file must be a ".mat" file, and the masks are saved as variable "FinalMasks". 
        dir_traces (str): The folder to save the unmixed traces.
        list_alpha (list of float, default to [0]): A list of alpha to be tested.
            The elements should be sorted in ascending order.
        Qclip (float, default to 0): Traces lower than this quantile are clipped to this quantile value.
            Qclip = 0 means no clipping is applied. 
        th_pertmin (float, default to 1): Maximum pertentage of unmixed traces equaling to the trace minimum.
            th_pertmin = 1 means no requirement is applied. 
        epsilon (float, default to 0): The minimum value of the input traces after scaling and shifting. 
        th_residual (float, default to 0): If not zero, The redisual of unmixing should be smaller than this value.
        nbin (int, default to 1): The temporal downsampling ratio.
            nbin = 1 means temporal downsampling is not used.
        bin_option (str, can be 'downsample' (default), 'sum', or 'mean'): 
            The method of temporal downsampling. 
            'downsample' means keep one frame and discard "nbin" - 1 frames for every "nbin" frames.
            'sum' means each binned frame is the sum of continuous "nbin" frames.
            'mean' means each binned frame is the mean of continuous "nbin" frames.
        multi_alpha (bool, default to True): 
            The approach to take when "list_alpha" has multiple elements.
            True means each element will be tested and saved independently.
            False means the largest element providing non-trivial output traces 
            will be used, which can be differnt for different neurons.
        flexible_alpha (bool, default to True): Whether a flexible alpha strategy is used 
            when the smallest alpha in "list_alpha" already caused over-regularization.
            False means the final alpha is the smallest element in "list_alpha".
            True means trying to recursively divide the smallest alpha by 2 until no over-regularization exists.

    Outputs:
        traces_nmfdemix (numpy.ndarray of float, shape = (T,n)): The resulting unmixed traces. 
            Each column is the unmixed trace of a neuron.
        list_mixout (list of numpy.ndarray of float, shape = (n1,n1)): 
            Each element is the row-normalized mixing matrix for the NMF of each neuron.

    In addition to the returned variables, more outputs are stored under the folder "dir_traces".
        There are two sub-folders under this folder.
        The sub-folder "raw" stores the neuron traces and the background traces before NMF unmixing.
        The sub-folder "alpha={}" stores the same "traces_nmfdemix" and "list_mixout", 
        as well as three other quantities characterizing the NMF unmixing process:
            list_alpha_final (list of float): Each element is the final chosen alpha for the NMF of each neuron. 
                It might be one of the elements in "list_alpha", or a value smaller than the first element.
            list_MSE (list of numpy.ndarray of float, shape = (n1,)): 
                Each element is the mean squared error (NMF residual) between 
                the input traces and the NMF-reconstructed traces for the NMF of each neuron.
            list_n_iter (list of int): Each element is the number of iterations 
                to achieve NMF convergence for the NMF of each neuron.
    '''

    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 

    start = time.time()
    # Get information about the video
    file_video = h5py.File(filename_video, 'r')
    varname = find_dataset(file_video)
    (T, Lx, Ly) = video_shape = file_video[varname].shape
    video_dtype = file_video[varname].dtype

    # Load neuron masks
    try:
        file_masks = loadmat(filename_masks)
        Masks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
    except:
        file_masks = h5py.File(filename_masks, 'r')
        Masks = np.array(file_masks['FinalMasks']).astype('bool')
        file_masks.close()
    (_, Lxm, Lym) = Masks.shape
    # If the shape of the masks is different from the shapes of the video,
    # zero-pad or crop the masks to fit the video shape 
    if Lxm < Lx:
        Masks = np.pad(Masks,((0,0),(0,Lx-Lxm),(0,0)),'constant', constant_values=0)
    elif Lxm > Lx:
        Masks = Masks[:,:Lx,:]
    if Lym < Ly:
        Masks = np.pad(Masks,((0,0),(0,0),(0,Ly-Lym)),'constant', constant_values=0)
    elif Lym > Ly:
        Masks = Masks[:,:,:Ly]
    masks_shape = Masks.shape

    # Determine the method to calculate traces: shared memory, memory mapping, or numpy
    if Masks.sum()*T < 7e7:
        trace_method = 'numba' # Using numba is faster for small videos
    else:
        if (sys.version_info.major+sys.version_info.minor/10)>=3.8:
            trace_method = 'shm' # "shared_memory" module is only available on python versions >= 3.8
        else:
            trace_method = 'memmap' # memory mapping is available on older python versions
        
    if trace_method == 'shm':
        # Create the shared memory object for the video
        nbytes_video = int(video_dtype.itemsize * file_video[varname].size)
        shm_video = SharedMemory(create=True, size=nbytes_video)
        video = np.frombuffer(shm_video.buf, dtype = file_video[varname].dtype)
        video[:] = file_video[varname][()].ravel()
        video = video.reshape(file_video[varname].shape)

        # Create the shared memory object for the masks
        shm_masks = SharedMemory(create=True, size=Masks.nbytes)
        FinalMasks = np.frombuffer(shm_masks.buf, dtype = 'bool')
        FinalMasks[:] = Masks.ravel()
        FinalMasks = FinalMasks.reshape(Masks.shape)
        del Masks

    elif trace_method == 'memmap':
        name_mmap = Exp_ID
        # Create the memory mapping file for the video
        fn_video = name_mmap + 'video.dat'
        fp_video = np.memmap(fn_video, dtype=video_dtype, mode='w+', shape=(T,Lx*Ly))
        for tt in range(T):
            fp_video[tt] = file_video[varname][tt].ravel()

        # Create the memory mapping file for the masks
        fn_masks = name_mmap + 'masks.dat'
        fp_masks = np.memmap(fn_masks, dtype='bool', mode='w+', shape=masks_shape)
        fp_masks[:] = Masks[:]
        del Masks

    elif trace_method == 'numba':
        video = np.array(file_video[varname])

        
    file_video.close()
    finish = time.time()
    print('Data loading time: {} s'.format(finish - start))

    # Calculate traces, background, and outside traces
    start = time.time()
    if trace_method == 'shm':
        (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_shm_neighbors(
            shm_video, video_dtype, video_shape, shm_masks, masks_shape, FinalMasks)
    elif trace_method == 'memmap':
        (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_mmap_neighbors(
            fn_video, video_dtype, video_shape, fn_masks, masks_shape, fp_masks, name_mmap)
    elif trace_method == 'numba':
        (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_numba_neighbors(
            video, Masks)

    # Save the raw traces into a ".mat" file under folder "dir_trace_raw".
    dir_trace_raw = os.path.join(dir_traces, "raw")
    finish = time.time()
    print('Trace calculation time: {} s'.format(finish - start))
    if not os.path.exists(dir_trace_raw):
        os.makedirs(dir_trace_raw)        
    savemat(os.path.join(dir_trace_raw, Exp_ID+".mat"), {"traces": traces, "bgtraces": bgtraces})

    if not isinstance(list_alpha, list):
        list_alpha = [list_alpha]
    else:
        list_alpha.sort()

    if multi_alpha:
        for alpha in list_alpha:
            # Apply NMF to unmix the background-subtracted traces
            start = time.time()
            traces_nmfdemix, list_mixout, list_MSE, list_final_alpha, list_n_iter = \
                use_nmfunmix(traces, bgtraces, outtraces, list_neighbors, [alpha], Qclip, \
                th_pertmin, epsilon, th_residual, nbin, bin_option, flexible_alpha)
            finish = time.time()
            print('NMF unmixing time: {} s'.format(finish - start))

            # Save the unmixed traces into a ".mat" file under folder "dir_trace_unmix".
            dir_trace_unmix = os.path.join(dir_traces, "alpha={:6.3f}".format(alpha))
            if not os.path.exists(dir_trace_unmix):
                os.makedirs(dir_trace_unmix)        
            savemat(os.path.join(dir_trace_unmix, Exp_ID+".mat"), {"traces_nmfdemix": traces_nmfdemix,\
                "list_mixout":list_mixout, "list_MSE":list_MSE, "list_final_alpha":list_final_alpha, "list_n_iter":list_n_iter})

    else:
        # Apply NMF to unmix the background-subtracted traces
        start = time.time()
        traces_nmfdemix, list_mixout, list_MSE, list_final_alpha, list_n_iter = \
            use_nmfunmix(traces, bgtraces, outtraces, list_neighbors, list_alpha, Qclip, \
            th_pertmin, epsilon, th_residual, nbin, bin_option, flexible_alpha)
        finish = time.time()
        print('Unmixing time: {} s'.format(finish - start))

        # Save the unmixed traces into a ".mat" file under folder "dir_trace_unmix".
        if len(list_alpha) > 1:
            dir_trace_unmix = os.path.join(dir_traces, "list_alpha={}".format(str(list_alpha)))
        else:
            dir_trace_unmix = os.path.join(dir_traces, "alpha={:6.3f}".format(list_alpha[0]))
        if not os.path.exists(dir_trace_unmix):
            os.makedirs(dir_trace_unmix)        
        savemat(os.path.join(dir_trace_unmix, Exp_ID+".mat"), {"traces_nmfdemix": traces_nmfdemix,\
            "list_mixout":list_mixout, "list_MSE":list_MSE, "list_final_alpha":list_final_alpha, "list_n_iter":list_n_iter})


    if trace_method == 'shm':
        # Unlink shared memory objects
        shm_video.close()
        shm_video.unlink()
        shm_masks.close()
        shm_masks.unlink()
    elif trace_method == 'memmap':
        # Delete memory mapping files
        fp_masks._mmap.close()
        del fp_masks
        os.remove(fn_masks)
        fp_video._mmap.close()
        del fp_video
        os.remove(fn_video)

    return traces_nmfdemix, list_mixout

