import numpy as np


def bin_trace(trace, nbin=1, bin_option='downsample'):
    ''' Unmix the input traces in Ftmix using NMF, and obtain the unmixed traces and the mixing matrix. 
    Inputs: 
        trace (numpy.ndarray of float, shape = (T,n)): The input unbinned traces.
            Each column represents an input temporal trace.
        nbin (int, default to 1): The temporal downsampling ratio.
            nbin = 1 means temporal downsampling is not used.
        bin_option (str, can be 'downsample' (default), 'sum', or 'mean'): 
            The method of temporal downsampling. 
            'downsample' means keep one frame and discard "nbin" - 1 frames for every "nbin" frames.
            'sum' means each binned frame is the sum of continuous "nbin" frames.
            'mean' means each binned frame is the mean of continuous "nbin" frames.

    Outputs:
        trace_bin (numpy.ndarray of float, shape = (T//nbin,n)): The resulting binned traces. 
            Each column is the binned trace corresponding to each column in trace.
    '''

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
            raise ValueError("bin_option must be 'downsample', 'sum', or 'mean'")

    return trace_bin
    

def bin_video(video, nbin=1, bin_option='downsample'):
    ''' Unmix the input traces in Ftmix using NMF, and obtain the unmixed traces and the mixing matrix. 
    Inputs: 
        video (numpy.ndarray of float, shape = (T,Lx,Ly)): The input unbinned video.
        nbin (int, default to 1): The temporal downsampling ratio.
            nbin = 1 means temporal downsampling is not used.
        bin_option (str, can be 'downsample' (default), 'sum', or 'mean'): 
            The method of temporal downsampling. 
            'downsample' means keep one frame and discard "nbin" - 1 frames for every "nbin" frames.
            'sum' means each binned frame is the sum of continuous "nbin" frames.
            'mean' means each binned frame is the mean of continuous "nbin" frames.

    Outputs:
        video_bin (numpy.ndarray of float, shape = (T//nbin,Lx,Ly)): The resulting binned video. 
    '''

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
            raise ValueError("bin_option must be 'downsample', 'sum', or 'mean'")

    return video_bin

