from scipy import special
import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)
np.seterr(divide='ignore',invalid='ignore')
from sklearn.metrics import mean_squared_error

from tuncat.bin_trace_video import bin_trace


def nmfunmix(Ftmix, nbin=1, tol=1e-4, max_iter=20000, alpha=1, l1_ratio=0.5, epsilon=0, bin_option='downsample'):
    ''' Unmix the input traces in Ftmix using NMF, and obtain the unmixed traces and the mixing matrix. 
    Inputs: 
        Ftmix (numpy.ndarray of float, shape = (T,n)): The input mixed traces.
            Each column represents an input temporal trace, which can be a trace of F, dF/F, or SNR.
            The first column (Ftmix[:,0]) is the time series of the neuron of interest.
            The last column (Ftmix[:,-1]) should be the time series of the outside activities
            - the outside activities should be the average of the pixels around the neuron of interest, 
            and these pixels should not be in the masks of the neighboring neuorns included in "Ftmix".
            Each of the remaining columns (Ftmix[:,1:-1]), if any, should be the time series of a neighboring neuron.
        nbin (int, default to 1): The temporal downsampling ratio.
            nbin = 1 means temporal downsampling is not used.
        tol (float, default to 1e-4): Tolerance of the stopping condition in NMF.
        max_iter (int, default to 20000): Maximum number of iterations before timing out in NMF.
        alpha (float, default to 0): The regularization parameter, 
            or the weight of the regularization term in the cost function. 
            alpha = 0 means no regularization, and should lead to a trivial solution (traceout = Ftmix).
            Larger alpha means higher regularization, which can cause shorter processing time 
            and more seperated signals, but if the alpha is too large, it can cause merged activity 
            from multiple neurons, a flat baseline, or even an output of identically zero.
        l1_ratio (float, 0 <= l1_ratio <= 1, default to 0.5): 
            The ratio controling the weight between the Frobenius norm and the L1 norm. 
            The panelty term is (1-l1_ratio) * Frobenius_norm + l1_ratio * L1_norm.
            For l1_ratio = 0, the penalty is a pure Frobenius norm.
            For l1_ratio = 1, the penalty is a pure L1 norm.
            For 0 < l1_ratio < 1, the penalty is a combination of L1 and Frobenius norms.
        epsilon (float, default to 0): The minimum value of the input traces after scaling and shifting. 
        bin_option (str, can be 'downsample' (default), 'sum', or 'mean'): 
            The method of temporal downsampling. 
            'downsample' means keep one frame and discard "nbin" - 1 frames for every "nbin" frames.
            'sum' means each binned frame is the sum of continuous "nbin" frames.
            'mean' means each binned frame is the mean of continuous "nbin" frames.

    Outputs:
        traceout (numpy.ndarray of float, shape = (T,n)): The resulting unmixed traces. 
            Each column is the unmixed trace corresponding to each column in Ftmix.
        mixout (numpy.ndarray of float, shape = (n,n)): The row-normalized mixing matrix.
        tempmixIDs (numpy.ndarray of float, binary value, shape = (n,n)):
            The assistent transformation matrix used to match raw output traces to the input traces.
        subtraces (list of int): The index of identically zero output traces.
            Idealy should be empty.
        MSE (numpy.ndarray of float, shape = (n,)): 
            Mean squared error (NMF residual) between the input traces and the NMF-reconstructed traces.
        n_iter (int): Number of iterations to achieve NMF convergence.

    Note:
        The matrices in the code are often the transposes of the corresponding matrices in the papre.
            Ftmix is the transpose of F_meas,
            Ftdemix is the transpose of F_sep,
            mixout is the transpose of M,
            tempmixIDs is the transpose of P.

    '''

    (T, n) = Ftmix.shape # T is the number of frames, n is the number of components

    # Normalize and shift the traces
    medFt = np.median(Ftmix, 0)
    minFt = Ftmix.min() 
    Ftmix = Ftmix - minFt
    q12 = np.quantile(Ftmix[:,0], [0.25, 0.5], axis=0) 
    noise = (q12[1]-q12[0])/(np.sqrt(2)*special.erfinv(0.5))
    Ftmix = Ftmix/noise + epsilon

    # This is the NMF call
    if nbin == 1: # not use temporal downsampling
        Ftmix_bin = Ftmix
    else: # use temporal downsampling
        Ftmix_bin = bin_trace(Ftmix, nbin, bin_option)

    # Initialize NMF
    if n <= min(Ftmix_bin.shape):
        init = 'nndsvdar'
    else:
        init = 'random'
    nmf = NMF(init=init, n_components=n, tol=tol, max_iter=max_iter,
              alpha=alpha, l1_ratio=l1_ratio)
    
    # NMF fit
    Ftdemix_bin = nmf.fit_transform(Ftmix_bin)
    mixout = nmf.components_
    n_iter = nmf.n_iter_
    if np.any(mixout):
        if nbin == 1:
            Ftdemix = Ftdemix_bin
        else: 
            # if using temporal downsampling, use the fitted mixing matrix 
            # to unmixed the input traces in the origianl temporal resolution
            Ftdemix = nmf.transform(Ftmix)
        Ftmix_guess = nmf.inverse_transform(Ftdemix) # NMF-reconstructed traces
        # Ftmix ~= Ftmix_guess = Ftdemix.dot(mixout)
    else: # Extreme over-regularization caused all output traces to be identically zero
        print('NMF not found. Please try with lower alpha.')
        traceout = np.zeros_like(Ftmix)
        mixout = np.zeros((n, n))
        tempmixIDs = np.zeros((n, n))
        subtraces = np.arange(n)
        MSE = np.zeros((n), dtype = Ftmix.dtype)
        return traceout, mixout, tempmixIDs, subtraces, MSE, n_iter

    # Assign output traces to input traces
    mixout_sum = mixout.sum(1)[:,np.newaxis] 
    num_outtraces = mixout_sum.nonzero()[0].size

    Ftdemix = Ftdemix * mixout_sum.T
    mixout = mixout / mixout_sum  # normalize mixout by each row
    mixout[np.isnan(mixout)] = 0

    # Iteratively assgin outputs according to the maximum element of normalized mixout
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

    # reorder and scale the outputs to match the inputs
    traceout = Ftdemix.dot(tempmixIDs * mixout)
    mixout = tempmixIDs.T.dot(mixout)  # reorder the mixing matrix
    # mixout = mixout / np.diag(mixout)[:,np.newaxis]
    subtraces = (traceout.sum(0) == 0).nonzero()[0]
    if subtraces.size:
        # If any output trace is identically zero, use the NMF residual as the output trace,
        # which is more informative than identically zero. 
        traceout[:, subtraces] = Ftmix[:, subtraces] - Ftmix_guess[:, subtraces]
    traceout = (traceout-epsilon) * noise + minFt  # Rescale the output traces to the actual inputs
    # Align the medians of the outputs to the inputs
    traceout = traceout - np.median(traceout, 0) + medFt
    MSE = mean_squared_error(Ftmix, Ftmix_guess, multioutput='raw_values')

    return traceout, mixout, tempmixIDs, subtraces, MSE, n_iter
