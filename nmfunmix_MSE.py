from scipy import special
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)
np.seterr(divide='ignore',invalid='ignore')
from sklearn.metrics import mean_squared_error

from bin_trace_video import bin_trace


# import line_profiler
# # profile = line_profiler.LineProfiler()
# @profile
def nmfunmix(Ftmix, nbin=1, tol=1e-4, max_iter=20000, alpha=1, l1_ratio=0.5, epsilon=0, bin_option='sum', doesplot=False):
    '''requires you to install scikit-learning [https://scikit-learn.org/stable/]
    with python 3.x+

    Ftmix - the columns of Ftmix enumerate the neurons, while the rows enumerate the time points.
    In this version, it can be traces of F, dF/F, or SNR.
    The first column is the time series of the neuron of interest
    The last but one column should be the time series of the outside activities - the outside activities
        should be the average of the pixels around the neuron of interest, and these pixels should
        not be in the masks of the neuorns in Ftmix.
    The last column should be the time series of the background activities - the background activities
        should be the median of all the pixels around the neuron of interest, including those
        inside and outside the masks of the neuorns in Ftmix.
    Adding the last background column is optional for two photon data, but is very helpful for one photon data.

    alpha - the overall sparseness regularizer (default 1). Increasing alpha
    will reduce the number of signals and/or the amplitudes of the transients
    that are in the unmixed set traces

    l1_ratio - the ratio between the Frobenius norm and the element-wise norm
    (default 0.5). A value of 1 for this input will make everything subject
    to the l1 norm (number of signals and number of positive amplitude values
    in each signal).

    doesplot - boolean value controling whether the traces are plotted (default false).

    traceout - is the resulting demixed traces, such that they have the same
    minimum and maximum as the original traces. For example traceout[:,i]
    would have the same minimum and maximum as Ftmix[:,i]. 
    traceout[:,0] is the unmixed trace of the neuron of interest.

    mixout - is the normalized mixingmatrix if many negative numbers appear
    here, then the unmixing probably went wrong

    tempmixIDs - is the elementary transformation matrix used to assign the raw output
    traces to the input traces.

    subtraces - is the index of not unmixed traces.

    The output graph is formatted as follows:
    black trace - original inputs
    red trace - traces that were unmixed
    green traces - traces that were not unmixed the result is a best
    estimate that subtracts the background
    '''

    ##
    (T, n) = Ftmix.shape
    # epsilon = 0 #  0.1 # 

    # linearly scale the traces
    medFt = np.median(Ftmix, 0)
    # minFt = Ftmix.min(axis=0) 
    minFt = Ftmix.min() 
    # minFt = np.quantile(Ftmix, 0.08, axis=0) 
    # minFt = medFt 
    Ftmix = Ftmix - minFt
    # np.clip(Ftmix, 0, None, out=Ftmix)
    # Ftmix[Ftmix<0] = 0
    # max90 = np.quantile(Ftmix, 0.9, axis=0).max() # max90%
    # max90 = np.quantile(Ftmix[:,0], 0.9, axis=0) # 90%
    # max90 = Ftmix[:,0].max() # max1
    # max90 = Ftmix.max() # max
    # max90 = np.median(Ftmix) # median_all
    # q12 = np.quantile(Ftmix, [0.25, 0.5], axis=0) 
    # max90 = (q12[1,:]-q12[0,:])/(np.sqrt(2)*special.erfinv(0.5)) # sigma
    q12 = np.quantile(Ftmix[:,0], [0.25, 0.5], axis=0) 
    max90 = (q12[1]-q12[0])/(np.sqrt(2)*special.erfinv(0.5)) # sigma1
    Ftmix = Ftmix/max90 + epsilon

    # This is the NMF call
    if nbin == 1:
        Ftmix_bin = Ftmix
    else:
        Ftmix_bin = bin_trace(Ftmix, nbin, bin_option)

    # setup to call NMF
    if n <= min(Ftmix_bin.shape):
        init = 'nndsvdar'
    else:
        init = 'random'
    nmf = NMF(init=init, n_components=n,
              tol=tol, # 1e-4,  # 1e-6, #
              max_iter=max_iter, # 1000,  # 1000, #
              alpha=alpha, l1_ratio=l1_ratio)
    
    Ftdemix_bin = nmf.fit_transform(Ftmix_bin)
    # Ftdemix_model = nmf.fit(Ftmix_bin)
    mixout = nmf.components_
    n_iter = nmf.n_iter_
    if np.any(mixout):
        if nbin == 1:
            Ftdemix = Ftdemix_bin
        else:
            Ftdemix = nmf.transform(Ftmix)
        Ftmix_guess = nmf.inverse_transform(Ftdemix)
    else:
        print('NMF not found. Please try with lower alpha.')
        traceout = np.zeros_like(Ftmix)
        mixout = np.zeros((n, n))
        tempmixIDs = np.zeros((n, n))
        subtraces = np.arange(n)
        # MSE = np.zeros((1, n), dtype = Ftmix.dtype)
        MSE = np.zeros((n), dtype = Ftmix.dtype)
        return traceout, mixout, tempmixIDs, subtraces, MSE, n_iter

    # Assign output traces to input traces
    mixout_sum = mixout.sum(1)[:,np.newaxis] 
    num_outtraces = mixout_sum.nonzero()[0].size

    Ftdemix = Ftdemix * mixout_sum.T
    mixout = mixout / mixout_sum  # normalize mixout by each row
    mixout[np.isnan(mixout)] = 0

    # iteratively assgin outputs according to the maximum element of normalized mixout
    tempmixIDs = np.zeros((n, n))
    tempmixout = mixout.copy()
    for _ in range(num_outtraces):
        # indmax = np.nanargmax(np.abs(tempmixout))
        # indmax1 = indmax // n
        # indmax2 = indmax % n
        if np.all(np.isnan(tempmixout)):
            continue
        (indmax1, indmax2) = (np.nanmax(tempmixout)==tempmixout).nonzero()
        tempmixIDs[indmax1, indmax2] = 1
        tempmixout[indmax1, : ] = 0
        tempmixout[: , indmax2] = 0
        tempmixout = tempmixout / tempmixout.sum(1)[:,np.newaxis]

    # mixout = tempmixIDs.T.dot(mixout)  # reorder the mixing matrix
    # reorder and scale the outputs to match the inputs
    # traceout = Ftdemix.dot(tempmixIDs * np.diag(mixout)[np.newaxis,:])
    traceout = Ftdemix.dot(tempmixIDs * mixout)
    mixout = tempmixIDs.T.dot(mixout)  # reorder the mixing matrix
    # mixout = mixout / mixout.sum(1)
    # mixout[np.isnan(mixout)] = 0
    subtraces = (traceout.sum(0) == 0).nonzero()[0]
    if subtraces.size:
        traceout[:, subtraces] = Ftmix[:, subtraces]-Ftmix_guess[:, subtraces]
    traceout = (traceout-epsilon) * max90 + minFt  # rescale the output traces to the actual inputs
    # align the medians of the outputs to the inputs
    traceout = traceout - np.median(traceout, 0) + medFt
    MSE = mean_squared_error(Ftmix, Ftmix_guess, multioutput='raw_values')

    return traceout, mixout, tempmixIDs, subtraces, MSE, n_iter
