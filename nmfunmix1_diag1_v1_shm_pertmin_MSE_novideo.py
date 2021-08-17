import numpy as np
from nmfunmix_MSE import nmfunmix
from scipy.io import savemat
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
# from r_neuropil import estimate_contamination_ratios

# import line_profiler
# # profile = line_profiler.LineProfiler()
# @profile
def nmfunmix1(i, trace, outtrace, alpha, \
        th_pertmin=1, epsilon=0, use_direction=False, nbin=1, bin_option='sum'):
    # th_pertmin = 1 # 0.16 # 
    # epsilon = 0
    n0 = 1
    eps = np.finfo(np.float32).eps
    # (T, n) = trace.shape
    # (Lx, Ly) = dims2

    # questionmasks = 0
    # omitmasks = []
    # find all cells near the center of the neuron of interest
    # r = np.sqrt((comx[i]-comx) **2 + (comy[i]-comy) **2)
    # neighbors = np.concatenate([np.array([i]), np.logical_and(r > 0, r < r_bg).nonzero()[0]])

    # r_dict = estimate_contamination_ratios(outtrace[:,np.newaxis], bgtrace[:,np.newaxis])
    # r = r_dict['r']
    # outtrace_clip = outtrace[:,np.newaxis] - r* bgtrace[:,np.newaxis]
    # outtrace_clip = outtrace[:,np.newaxis]-bgtrace[:,np.newaxis]
    # if Qclip>0:
    #     np.clip(outtrace_clip, np.quantile(outtrace_clip, Qclip, axis=0), None, out=outtrace_clip)

    # call nmfunmix
    # gradually increase alpha until the output mixout is singular,
    # then choose the maximum alpha that provides nonsingular mixout
    tracein = np.concatenate([trace, outtrace[:,np.newaxis]], axis=1) # [:, neighbors]
    for j in range(len(alpha)):
        (traceout, mixout, tempmixIDs, subtraces, MSE, n_iter) = nmfunmix(tracein, nbin=nbin, alpha=alpha[j], epsilon=epsilon, bin_option=bin_option) 
        alpha_final = alpha[j]
        # std_mixout = np.std(mixout, axis=1)
        # std_mixout[subtraces] = np.nan
        # bgind = np.nanargmin(std_mixout)
        if th_pertmin < 1:
            pertmin = (np.abs(traceout[:,0] - traceout[:,0].min())<eps).mean() > th_pertmin
        else: 
            pertmin = False
        if use_direction:
            # negative = np.median(traceout[:,0]) > traceout[:,0].mean()
            negative = 2 * np.median(traceout[:,0]) > traceout[:,0].max() + traceout[:,0].min()
        else:
            negative = False
        question_flag = pertmin or subtraces.size or negative   # and (subtraces[0]<neighbors.size or subtraces.size>2) # or bgind<neighbors.size
        # If there is a subtrace and it is not assigned to either the
        # background or the outside trace, or there are two subtraces,
        # then return to the privous alpha
        if question_flag: # False: # 
            jj = j*1
            if jj > 0:
                while jj > 0 and question_flag:
                    jj = jj-1
                    (traceout, mixout, tempmixIDs, subtraces, MSE, n_iter) = nmfunmix(tracein, nbin=nbin, alpha=alpha[jj], epsilon=epsilon, bin_option=bin_option) 
                    alpha_final = alpha[jj]
                    # std_mixout = np.std(mixout, axis=1)
                    # std_mixout[subtraces] = np.nan
                    # bgind = np.nanargmin(std_mixout)
                    if th_pertmin < 1:
                        pertmin = (np.abs(traceout[:,0] - traceout[:,0].min())<eps).mean() > th_pertmin
                    else: 
                        pertmin = False
                    if use_direction:
                        # negative = np.median(traceout[:,0]) > traceout[:,0].mean()
                        negative = 2 * np.median(traceout[:,0]) > traceout[:,0].max() + traceout[:,0].min()
                    else:
                        negative = False
                    question_flag = pertmin or subtraces.size or negative   # and (subtraces[0]<neighbors.size or subtraces.size>2) # or bgind<neighbors.size
            if jj == 0:  # if the first alphas is already too large, then iteratively divide it by 2.
                alpha_temp = alpha[0]
                while question_flag and alpha_temp>alpha[0]/5: # 1e-4: # 
                    alpha_temp = alpha_temp/2
                    (traceout, mixout, tempmixIDs, subtraces, MSE, n_iter) = nmfunmix(tracein, nbin=nbin, alpha=alpha_temp, epsilon=epsilon, bin_option=bin_option) 
                    alpha_final = alpha_temp
                    # std_mixout = np.std(mixout, axis=1)
                    # std_mixout[subtraces] = np.nan
                    # bgind = np.nanargmin(std_mixout)
                    if th_pertmin < 1:
                        pertmin = (np.abs(traceout[:,0] - traceout[:,0].min())<eps).mean() > th_pertmin
                    else: 
                        pertmin = False
                    if use_direction:
                        # negative = np.median(traceout[:,0]) > traceout[:,0].mean()
                        negative = 2 * np.median(traceout[:,0]) > traceout[:,0].max() + traceout[:,0].min()
                    else:
                        negative = False
                    question_flag = pertmin or subtraces.size or negative   # and (subtraces[0]<neighbors.size or subtraces.size>2) # or bgind<neighbors.size
            break

    print('finished neuron', i)
    return traceout, mixout, outtrace, tempmixIDs, subtraces, alpha_final, MSE, tracein, n_iter
    # return traceout, mixout, neighbors, outtrace, tempmixIDs, subtraces, omitmasks, questionmasks, alpha_final, MSE

