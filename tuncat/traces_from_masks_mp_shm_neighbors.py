import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory


def mean_median_out_trace(nn, shm_video, video_dtype, video_shape, \
    shm_masks, shm_xx, shm_yy, r_bg, comx, comy, area, neighbors):
    ''' Calculate the traces of the neuron, background, and outside region for a neuron. 
    Inputs: 
        nn (int): The index of the neuron of interest.
        shm_video (SharedMemory object): Shared memory storing the input video.
        video_dtype (str): The data type of the input video.
        video_shape (tuple of int, shape = (3,)): The shape of the input video.
        shm_masks (SharedMemory object): Shared memory storing the binary spatial masks of all neurons.
        shm_xx (SharedMemory object): Shared memory storing the x position of a grid.
        shm_yy (SharedMemory object): Shared memory storing the y position of a grid.
        r_bg (float): The radius of the background mask.
        comx (numpy.ndarray of float, shape = (n,)): The x-position of the center of all masks.
        comy (numpy.ndarray of float, shape = (n,)): The y-position of the center of all masks.
        area (int): The area of the mask.
        neighbors (list of int): A list of indeces of neighboring neurons.

    Outputs:
        trace (numpy.ndarray of float, shape = (T,)): The raw trace of the neuron.
        bgtrace (numpy.ndarray of float, shape = (T,)): The raw background trace around the neuron.
        outtrace (numpy.ndarray of float, shape = (T,)): The raw trace of all the outside activities around the neuron.
    '''

    (T, Lx, Ly) = video_shape
    n = comx.size
    # Reconstruct the video, masks, and corrdinates from shared memory objects.
    video = np.ndarray((T, Lx*Ly), buffer=shm_video.buf, dtype = video_dtype)
    FinalMasks = np.ndarray((n, Lx, Ly), buffer=shm_masks.buf, dtype = 'bool')
    mask = FinalMasks[nn]
    xx = np.ndarray((Lx,Ly), buffer=shm_xx.buf, dtype='uint16')
    yy = np.ndarray((Lx,Ly), buffer=shm_yy.buf, dtype='uint16')

    # Expand the radius of the background mask if the neuron area is large.
    r_bg_0 = max(r_bg, np.sqrt(area/np.pi) * 2.5)
    circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_0 ** 2 # Background mask
    bgtrace = np.median(video[:, circleout.ravel()], 1) # Background trace
    trace = video[:, mask.ravel()].mean(1) # Neuron trace

    fgmask = FinalMasks[neighbors].sum(0) > 0
    r_bg_large = r_bg * 0.8
    circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_large ** 2
    bgmask = np.logical_and(circleout,np.logical_not(fgmask)) # Outside mask
    bgweight = bgmask.sum() # Area of outside mask
    # Adjust the radius of the outside mask, so that the area of the outside mask is larger than half of the neuron area
    while bgweight < area/2: # == 0: # 
        r_bg_large = r_bg_large+1
        circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_large ** 2
        bgmask = np.logical_and(circleout,np.logical_not(fgmask))
        bgweight = bgmask.sum()
    outtrace = video[:, bgmask.ravel()].mean(1) # Outside trace

    return trace, bgtrace, outtrace


def traces_bgtraces_from_masks_shm_neighbors(shm_video, video_dtype, video_shape, \
        shm_masks, masks_shape, FinalMasks):
    ''' Calculate the traces of the neuron, background, and outside region for all neurons. 
    Inputs: 
        shm_video (SharedMemory object): Shared memory storing the input video.
        video_dtype (str): The data type of the input video.
        video_shape (tuple of int, shape = (3,)): The shape of the input video.
        shm_masks (SharedMemory object): Shared memory storing the binary spatial masks of all neurons.
        masks_shape (tuple of int, shape = (3,)): The shape of the neuron masks.
        FinalMasks (numpy.ndarray of bool, shape = (n,Lx,Ly)): The binary spatial masks of all neurons.
            Each slice represents a binary mask of a neuron.

    Outputs:
        traces (numpy.ndarray of float, shape = (T,n)): The raw traces of all neurons.
            Each column represents a trace of a neuron.
        bgtraces (numpy.ndarray of float, shape = (T,n)): All the raw background traces.
            Each column represents a background trace corresponding to a neuron.
        outtraces (numpy.ndarray of float, shape = (T,n)): The raw traces of all the outside activities.
            Each column represents an outside trace corresponding to a neuron.
        list_neighbors (list of list of int): 
            Each element is a list of indeces of neighboring neurons of a neuron.
    '''

    (ncells, Lx, Ly) = masks_shape
    # Find the neighboring neurons of all neurons. 
    (list_neighbors, comx, comy, area, r_bg) = find_neighbors(FinalMasks)

    # Create shared memory objects for x and y coordinates
    [xx, yy] = np.meshgrid(np.arange(Ly), np.arange(Lx))
    xx = xx.astype('uint16')
    shm_xx = SharedMemory(create=True, size=xx.nbytes)
    xx_temp = np.ndarray(xx.shape, dtype='uint16', buffer=shm_xx.buf)
    xx_temp[:] = xx[:]
    # xx_temp = np.frombuffer(shm_xx.buf, dtype='uint16')
    # xx_temp[:] = xx.ravel()
    yy = yy.astype('uint16')
    shm_yy = SharedMemory(create=True, size=yy.nbytes)
    yy_temp = np.ndarray(yy.shape, dtype='uint16', buffer=shm_yy.buf)
    yy_temp[:] = yy[:]
    # yy_temp = np.frombuffer(shm_yy.buf, dtype='uint16')
    # yy_temp[:] = yy.ravel()
    
    # results = []
    # for nn in range(ncells):
    #     results.append(mean_median_out_trace(nn, shm_video, video_dtype, video_shape, \
    #         shm_masks, shm_xx, shm_yy, r_bg, comx, comy, area[nn], list_neighbors[nn]))

    # Calculate the traces of the neuron, background, and outside region for each neuron. 
    p = mp.Pool(mp.cpu_count())
    results = p.starmap(mean_median_out_trace, [(nn, shm_video, video_dtype, video_shape, \
        shm_masks, shm_xx, shm_yy, r_bg, comx, comy, area[nn], list_neighbors[nn]) for nn in range(ncells)], chunksize=1)
    p.close()

    traces = np.vstack([x[0] for x in results]).T
    bgtraces = np.vstack([x[1] for x in results]).T
    outtraces = np.array([x[2] for x in results]).T

    # Unlink shared memory objects
    del xx_temp
    shm_xx.close()
    shm_xx.unlink()
    del yy_temp
    shm_yy.close()
    shm_yy.unlink()

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
    area = np.zeros(n, dtype = 'uint32')
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

