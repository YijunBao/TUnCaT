from scipy import special
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)
np.seterr(divide='ignore',invalid='ignore')


def bin_trace(trace, nbin=1, bin_option='sum'):
    if nbin==1:
        trace_bin = trace
    else:
        (T,n) = trace.shape
        trace_bin = np.zeros((T//nbin,n), dtype = trace.dtype)
        if bin_option == 'sum':
            for t in range(T//nbin):
                trace_bin[t] = trace[t*nbin:(t+1)*nbin].sum(0)
        elif bin_option == 'mean':
            for t in range(T//nbin):
                trace_bin[t] = trace[t*nbin:(t+1)*nbin].mean(0)
        elif bin_option == 'downsample':
            for t in range(T//nbin):
                trace_bin[t] = trace[t*nbin]
        else:
            raise ValueError("bin_option must be 'sum' or 'mean'")

    return trace_bin
    # return trace[::nbin]
    

def bin_video(video, nbin=1, bin_option='mean'):
    if nbin==1:
        video_bin = video
    else:
        (T,Lx,Ly) = video.shape
        video_bin = np.zeros((T//nbin,Lx,Ly), dtype = video.dtype)
        if bin_option == 'sum':
            for t in range(T//nbin):
                video_bin[t] = video[t*nbin:(t+1)*nbin].sum(0)
        elif bin_option == 'mean':
            for t in range(T//nbin):
                video_bin[t] = video[t*nbin:(t+1)*nbin].mean(0)
        elif bin_option == 'downsample':
            for t in range(T//nbin):
                video_bin[t] = video[t*nbin]
        else:
            raise ValueError("bin_option must be 'sum' or 'mean'")

    return video_bin
    # return video[::nbin]

