import numpy as np
import multiprocessing as mp
from scipy.io import savemat
from nmfunmix1_pertmin_res_MSE_novideo import nmfunmix1


def use_nmfunmix(traces, bgtraces, outtraces, list_neighbors, list_alpha=[0], Qclip=0, \
        th_pertmin=1, epsilon=0, th_residual=0, nbin=1, bin_option='downsample', flexible_alpha=True):
    ''' Unmix the traces of all neurons using NMF, and obtain the unmixed traces and the mixing matrix. 
    Inputs: 
        traces (numpy.ndarray of float, shape = (T,n)): The raw traces of all neurons.
            Each column represents a trace of a neuron, which can be a trace of F, dF/F, or SNR.
        bgtraces (numpy.ndarray of float, shape = (T,n)): All the raw background traces.
            Each column represents a background trace corresponding to a neuron.
        outtraces (numpy.ndarray of float, shape = (T,n)): The raw traces of all the outside activities.
            Each column represents an outside trace corresponding to a neuron.
        list_neighbors (list of list of int): 
            Each element is a list of indeces of neighboring neurons of a neuron.
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
        flexible_alpha (bool, default to True): Whether a flexible alpha strategy is used 
            when the smallest alpha in "list_alpha" already caused over-regularization.
            False means the final alpha is the smallest element in "list_alpha".
            True means trying to recursively divide the smallest alpha by 2 until no over-regularization exists.

    Outputs:
        demix (numpy.ndarray of float, shape = (T,n)): The resulting unmixed traces. 
            Each column is the unmixed trace of a neuron.
        list_mixout (list of numpy.ndarray of float, shape = (n1,n1)): 
            Each element is the row-normalized mixing matrix for the NMF of each neuron.
        list_alpha_final (list of float): Each element is the final chosen alpha for the NMF of each neuron. 
            It might be one of the elements in "list_alpha", or a value smaller than the first element.
        list_MSE (list of numpy.ndarray of float, shape = (n1,)): 
            Each element is the mean squared error (NMF residual) between 
            the input traces and the NMF-reconstructed traces for the NMF of each neuron.
        list_n_iter (list of int): Each element is the number of iterations 
            to achieve NMF convergence for the NMF of each neuron.
    '''

    n = traces.shape[1]
    # Background subtraction for neuron and outside traces
    traces_clip = traces-bgtraces
    outtrace_clip = outtraces-bgtraces
    if Qclip>0: # Clip the bottom "Qclip" quantile, to eliminate negative transients.
        np.clip(traces_clip, np.quantile(traces_clip, Qclip, axis=0), None, out=traces_clip)
        np.clip(outtrace_clip, np.quantile(outtrace_clip, Qclip, axis=0), None, out=outtrace_clip)

    # results = []
    # for i in range(n):
    #     result = nmfunmix1(i, traces_clip[:, list_neighbors[i]], bgtraces[:,i], list_alpha, \
    #         th_pertmin, epsilon, th_residual, nbin, bin_option)
    #     results.append(result)

    # Apply NMF to unmix each group of input traces corresponding to each neuron.
    p = mp.Pool(mp.cpu_count())
    results = p.starmap(nmfunmix1, [(i, traces_clip[:, list_neighbors[i]], outtrace_clip[:,i], list_alpha, 
        th_pertmin, epsilon, th_residual, nbin, bin_option, flexible_alpha) for i in range(n)], chunksize=1)
    p.close()

    # See the explanation of "nmfunmix1" for detailed explanation of these output quantities
    list_traceout = ([x[0].T for x in results])
    demix = np.concatenate([x[0:1] for x in list_traceout], 0).T

    list_mixout = ([x[1] for x in results])
    list_final_alpha = np.array([x[5] for x in results])
    list_MSE = [x[6] for x in results]
    list_n_iter = [x[8] for x in results]

    # list_outtrace = ([x[2] for x in results])
    # list_tempmixIDs = ([x[3] for x in results])
    # list_subtraces = ([x[4] for x in results])
    # list_tracein = ([x[7].T for x in results])

    # Can uncomment the previuos and following lines to save the unmixing results to a mat file,
        # mainly for debugging perpose.
    # savemat('demixtest_{}.mat'.format(alpha[0]), {'demix':demix, 'list_tracein':list_tracein, 'list_traceout':list_traceout,\
    #     'list_mixout':list_mixout, 'list_neighbors':list_neighbors, 'list_subtraces':list_subtraces, \
    #     'list_outtrace':list_outtrace, 'list_final_alpha':list_final_alpha, 'list_tempmixIDs':list_tempmixIDs})

    return demix, list_mixout, list_MSE, list_final_alpha, list_n_iter
