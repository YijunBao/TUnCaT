import numpy as np
from nmfunmix_MSE import nmfunmix


def nmfunmix1(i, trace, outtrace, list_alpha=[0], th_pertmin=1, epsilon=0, \
        use_direction=False, nbin=1, bin_option='downsample', flexible_alpha=True):
    ''' Unmix the input traces in "trace" and "outtrace" using NMF, and obtain the unmixed traces and the mixing matrix. 
    Inputs: 
        i (int): The index of the neuron of interest.
        trace (numpy.ndarray of float, shape = (T,n-1)): The background-subtracted traces for neurons,
            including the neuron of interest and neighboring neurons.
            Each column represents an input temporal trace, which can be a trace of F, dF/F, or SNR.
            The first column (trace[:,0]) is the time series of the neuron of interest.
            Each of the remaining columns (trace[:,1:]), if any, should be the time series of a neighboring neuron.
        outtrace (numpy.ndarray of float, shape = (T,)): The background-subtracted trace of the outside activities.
            - the outside activities should be the average of the pixels around the neuron of interest, 
            and these pixels should not be in the masks of the neighboring neuorns included in "trace".
        list_alpha (list of float, default to [0]): A list of alpha to be tested.
            The elements should be sorted in ascending order.
        th_pertmin (float, default to 1): Maximum pertentage of unmixed traces equaling to the trace minimum.
            th_pertmin = 1 means no requirement is applied. 
        epsilon (float, default to 0): The minimum value of the input traces after scaling and shifting. 
        use_direction (bool, default to False): Whether a direction requirement is applied to the output traces.
            A direction requirement means the positive transients should be farther away from baseline than negative transients.
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
        traceout (numpy.ndarray of float, shape = (T,n)): The resulting unmixed traces. 
            Each column is the unmixed trace corresponding to each column in tracein.
        mixout (numpy.ndarray of float, shape = (n,n)): The row-normalized mixing matrix.
        outtrace (numpy.ndarray of float, shape = (T,n)): Same as the input "outtrace".
        tempmixIDs (numpy.ndarray of float, binary value, shape = (n,n)):
            The assistent transformation matrix used to match raw output traces to the input traces.
        subtraces (list of int): The index of identically zero output traces.
            Idealy should be empty.
        alpha_final (float): The final chosen alpha. It might be one of the elements in "list_alpha", 
            or a value smaller than the first element.
        MSE (numpy.ndarray of float, shape = (n,)): 
            Mean squared error (NMF residual) between the input traces and the NMF-reconstructed traces.
        tracein (numpy.ndarray of float, shape = (T,n)): The input traces to NMF. 
            Each column represents an input temporal trace.
            The first column (tracein[:,0]) is the time series of the neuron of interest.
            The last column (tracein[:,-1]) should be the time series of the outside activities
            - the outside activities should be the average of the pixels around the neuron of interest, 
            and these pixels should not be in the masks of the neighboring neuorns included in "tracein".
            Each of the remaining columns (tracein[:,1:-1]), if any, should be the time series of a neighboring neuron.
        n_iter (int): Number of iterations to achieve NMF convergence.
    '''

    eps = np.finfo(np.float32).eps
    if not isinstance(list_alpha, list):
        list_alpha = [list_alpha]
    else:
        list_alpha.sort()

    tracein = np.concatenate([trace, outtrace[:,np.newaxis]], axis=1)
    # call nmfunmix
    # Gradually increase alpha until the output mixout is singular,
    # then choose the maximum alpha that provides nonsingular mixout
    for j in range(len(list_alpha)):
        (traceout, mixout, tempmixIDs, subtraces, MSE, n_iter) = nmfunmix(tracein, \
            nbin=nbin, alpha=list_alpha[j], epsilon=epsilon, bin_option=bin_option) 
        alpha_final = list_alpha[j]
        # "pertmin" is the indicator of whether the pertentage of unmixed traces equaling to the trace minimum exceeds "th_pertmin"
        if th_pertmin < 1:
            pertmin = (np.abs(traceout[:,0] - traceout[:,0].min())<eps).mean() > th_pertmin
        else: 
            pertmin = False
        # "negative" is the indicator of whether the positive transients are farther away from baseline than negative transients
        if use_direction:
            negative = 2 * np.median(traceout[:,0]) > traceout[:,0].max() + traceout[:,0].min()
        else:
            negative = False
        question_flag = pertmin or subtraces.size or negative
        
        # question_flag == True means the alpha is too large. 
        if question_flag:
            jj = j*1
            if jj > 0: # If the alpha is too large only at some late alpha, then return to the privous alpha
                while jj > 0 and question_flag:
                    jj = jj-1
                    (traceout, mixout, tempmixIDs, subtraces, MSE, n_iter) = nmfunmix(tracein, \
                        nbin=nbin, alpha=list_alpha[jj], epsilon=epsilon, bin_option=bin_option) 
                    alpha_final = list_alpha[jj]
                    if th_pertmin < 1:
                        pertmin = (np.abs(traceout[:,0] - traceout[:,0].min())<eps).mean() > th_pertmin
                    else: 
                        pertmin = False
                    if use_direction:
                        negative = 2 * np.median(traceout[:,0]) > traceout[:,0].max() + traceout[:,0].min()
                    else:
                        negative = False
                    question_flag = pertmin or subtraces.size or negative 
            if jj == 0 and flexible_alpha:  
                # if the first alphas already caused over-regularization, and flexible alpha strategy is used,
                # then recursively divide alpha by 2 until no over-regularization exists.
                alpha_temp = list_alpha[0]
                while question_flag and alpha_temp>list_alpha[0]/5: # 1e-4: # 
                    alpha_temp = alpha_temp/2
                    (traceout, mixout, tempmixIDs, subtraces, MSE, n_iter) = nmfunmix(tracein, \
                        nbin=nbin, alpha=alpha_temp, epsilon=epsilon, bin_option=bin_option) 
                    alpha_final = alpha_temp
                    if th_pertmin < 1:
                        pertmin = (np.abs(traceout[:,0] - traceout[:,0].min())<eps).mean() > th_pertmin
                    else: 
                        pertmin = False
                    if use_direction:
                        negative = 2 * np.median(traceout[:,0]) > traceout[:,0].max() + traceout[:,0].min()
                    else:
                        negative = False
                    question_flag = pertmin or subtraces.size or negative 
            break

    print('finished neuron', i)
    return traceout, mixout, outtrace, tempmixIDs, subtraces, alpha_final, MSE, tracein, n_iter

