import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from scipy.io import savemat
# from nmfunmix1_diag1_v1_shm_nodecrease_MSE import nmfunmix1
from nmfunmix1_diag1_v1_shm_pertmin_MSE_novideo import nmfunmix1
# from r_neuropil import estimate_contamination_ratios


def use_nmfunmix(traces, bgtraces, outtraces, list_neighbors, alpha, Qclip=0, \
        th_pertmin=1, epsilon=0, use_direction=False, nbin=1, bin_option='sum'):
    '''traces is the mixed traces (sum of values in mask) in (n, T) format n is index of neurons
    bgtraces is the background traces
    B is the movie (should make it in "double" format)
    FinalMasks are the masks in (x,y,n) format
    photons = 1 or 2 indicates whether the video is from one-photon or
     two-photon microscope. Default is 2.
    demix is the demixed traces
    Relative data are stored in "demixtest.mat".

    radius (r_bg) for finding nearby neurons this is found by taking the average of
    the neuron diamters assuming a circle shape and multiplying by 1.25 - 1.5
    comx and comy are the centroids of masks.
    '''

    # bgtraces = np.zeros_like(bgtraces)
    # parameter setting
    # (n, Lx, Ly) = masks_shape

    # r_dict = estimate_contamination_ratios(trace, bgtraces)
    # r = r_dict['r']
    n = traces.shape[1]
    # traces_clip = trace - r* bgtraces
    traces_clip = traces-bgtraces
    outtrace_clip = outtraces-bgtraces
    if Qclip>0:
        np.clip(traces_clip, np.quantile(traces_clip, Qclip, axis=0), None, out=traces_clip)
        np.clip(outtrace_clip, np.quantile(outtrace_clip, Qclip, axis=0), None, out=outtrace_clip)

    # results = []
    # for i in range(n):
    #     result = nmfunmix1(i, traces_clip[:, list_neighbors[i]], bgtraces[:,i], alpha, \
    #         th_pertmin, epsilon, use_direction, nbin, bin_option)
    #     results.append(result)

    p = mp.Pool(mp.cpu_count())
    results = p.starmap(nmfunmix1, [(i, traces_clip[:, list_neighbors[i]], outtrace_clip[:,i], alpha, 
        th_pertmin, epsilon, use_direction, nbin, bin_option) for i in range(n)], chunksize=1)
    p.close()

    list_traceout = ([x[0].T for x in results])
    demix = np.concatenate([x[0:1] for x in list_traceout], 0).T

    list_mixout = ([x[1] for x in results])
    # list_outtrace = ([x[2] for x in results])
    # list_tempmixIDs = ([x[3] for x in results])
    # list_subtraces = ([x[4] for x in results])
    # list_tracein = ([x[7].T for x in results])

    list_final_alpha = np.array([x[5] for x in results])
    list_MSE = [x[6] for x in results]
    list_n_iter = [x[8] for x in results]

    # savemat('demixtest_new.mat', {'demix':demix, 'list_tracein':list_tracein, 'list_traceout':list_traceout,\
    #     'list_mixout':list_mixout, 'list_neighbors':list_neighbors, 'list_subtraces':list_subtraces, \
    #     'list_outtrace':list_outtrace, 'list_final_alpha':list_final_alpha, 'list_tempmixIDs':list_tempmixIDs})

    # use_nmfunmix_refine('demixtest.mat')

    return demix, list_mixout, list_MSE, list_final_alpha, list_n_iter
