import numpy as np
from numba import jit, prange


@jit("void(u2[:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fast_mean_u2(video, trace):
    '''Calculate the means of each frame as the temporal traces
    Inputs: 
        video(numpy.ndarray of uint16, shape = (T,L)): The input video reshaped to 2D.
    Outputs:
        trace(numpy.ndarray of float32, shape = (L,)): The means of each frame.
    '''
    for tt in prange(video.shape[0]):
        trace[tt] = np.mean(video[tt, :])


@jit("void(u2[:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fast_median_u2(video, bgtrace):
    '''Calculate the medians of each frame as the temporal traces
    Inputs: 
        video(numpy.ndarray of uint16, shape = (T,L)): The input video reshaped to 2D.
    Outputs:
        bgtrace(numpy.ndarray of float32, shape = (L,)): The medians of each frame.
    '''
    for tt in prange(video.shape[0]):
        bgtrace[tt] = np.median(video[tt, :])


@jit("void(f4[:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fast_mean_f4(video, trace):
    '''Calculate the means of each frame as the temporal traces
    Inputs: 
        video(numpy.ndarray of float32, shape = (T,L)): The input video reshaped to 2D.
    Outputs:
        trace(numpy.ndarray of float32, shape = (L,)): The means of each frame.
    '''
    for tt in prange(video.shape[0]):
        trace[tt] = np.mean(video[tt, :])


@jit("void(f4[:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fast_median_f4(video, bgtrace):
    '''Calculate the medians of each frame as the temporal traces
    Inputs: 
        video(numpy.ndarray of float32, shape = (T,L)): The input video reshaped to 2D.
    Outputs:
        bgtrace(numpy.ndarray of float32, shape = (L,)): The medians of each frame.
    '''
    for tt in prange(video.shape[0]):
        bgtrace[tt] = np.median(video[tt, :])


def traces_bgtraces_from_masks_numba_neighbors(video, masks):
    ''' Calculate the traces of the neuron, background, and outside region for a neuron. 
    Inputs: 
        video (numpy.ndarray, shape = (T,Lx,Ly)): The input video.
        masks (numpy.ndarray of bool, shape = (n,Lx,Ly)): The binary spatial masks of all neurons.
            Each slice represents a binary mask of a neuron.

    Outputs:
        trace (numpy.ndarray of float32, shape = (T,)): The raw trace of the neuron.
        bgtrace (numpy.ndarray of float32, shape = (T,)): The raw background trace around the neuron.
        outtrace (numpy.ndarray of float32, shape = (T,)): The raw trace of all the outside activities around the neuron.
    '''

    (T, Lx, Ly) = video.shape
    (ncells, Lxm, Lym) = masks.shape

    # Find the neighboring neurons of all neurons. 
    (list_neighbors, comx, comy, area, r_bg) = find_neighbors(masks)
    [xx, yy] = np.meshgrid(np.arange(Ly), np.arange(Lx))
    xx = xx.astype('uint16')
    yy = yy.astype('uint16')

    # Reshape 3D video to 2D
    video = video.reshape((T, Lxm*Lym))

    traces = np.zeros((T, ncells), dtype='float32')
    bgtraces = np.zeros((T, ncells), dtype='float32')
    outtraces = np.zeros((T, ncells), dtype='float32')

    if video.dtype == 'uint16':
        fast_mean = fast_mean_u2
        fast_median = fast_median_u2
    elif video.dtype == 'float32':
        fast_mean = fast_mean_f4
        fast_median = fast_median_f4
    else:
        fast_mean = fast_mean_f4
        fast_median = fast_median_f4
        video = video.astype('float32')

    for nn in range(ncells):
        mask = masks[nn]
        # Expand the radius of the background mask if the neuron area is large.
        r_bg_0 = max(r_bg, np.sqrt(area[nn]/np.pi) * 2.5)
        circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_0 ** 2
        fast_median(video[:, circleout.ravel()], bgtraces[:, nn]) # Background trace
        fast_mean(video[:, mask.ravel()], traces[:, nn]) # Neuron trace

        fgmask = masks[list_neighbors[nn]].sum(0) > 0
        r_bg_large = r_bg * 0.8
        circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_large ** 2
        bgmask = np.logical_and(circleout,np.logical_not(fgmask)) # Outside mask
        bgweight = bgmask.sum()
        # Adjust the radius of the outside mask, so that the area of the outside mask is larger than half of the neuron area
        while bgweight < area[nn]/2: # == 0: # 
            r_bg_large = r_bg_large+1
            circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_large ** 2
            bgmask = np.logical_and(circleout,np.logical_not(fgmask))
            bgweight = bgmask.sum()
        fast_mean(video[:, bgmask.ravel()], outtraces[:, nn]) # Outside trace

    return traces, bgtraces, outtraces, list_neighbors


def find_neighbors(FinalMasks):
    ''' Find the neighboring neurons of all neurons. 
    Inputs: 
        FinalMasks (numpy.ndarray of bool, shape = (n,Lx,Ly)): The binary spatial masks of all neurons.
            Each slice represents a binary mask of a neuron.

    Outputs:
        list_neighbors (list of list of int): 
            Each element is a list of indeces of neighboring neurons of a neuron.
        comx (numpy.ndarray of float, shape = (n,)): The x-position of the center of each mask.
        comy (numpy.ndarray of float, shape = (n,)): The y-position of the center of each mask.
        area (numpy.ndarray of int, shape = (n,)): The area of each mask.
        r_bg (float): The radius of the background mask.
    '''

    # (n, Lx, Ly) = FinalMasks.shape
    n = FinalMasks.shape[0]
    list_neighbors = []
    area = np.zeros(n)
    comx = np.zeros(n)
    comy = np.zeros(n)
    for i in range(n):
        mask = FinalMasks[i]
        [xxs, yys] = mask.nonzero()
        area[i] = xxs.size
        comx[i] = xxs.mean()
        comy[i] = yys.mean()

    r_bg = np.sqrt(area.mean()/np.pi)*2.5
    for i in range(n):
        # The distance between each pair of neurons
        r = np.sqrt((comx[i]-comx) **2 + (comy[i]-comy) **2)
        # Neighboring neurons are defined as the neurons whose centers are 
        # less than "r_bg" away from the center of the neuron of interest. 
        neighbors = np.concatenate([np.array([i]), np.logical_and(r > 0, r < r_bg).nonzero()[0]])
        list_neighbors.append(neighbors)

    return list_neighbors, comx, comy, area, r_bg


