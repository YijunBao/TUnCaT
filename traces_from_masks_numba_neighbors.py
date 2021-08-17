import numpy as np
from numba import jit, prange


@jit("void(u2[:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fast_mean_u2(video, trace):
    for tt in prange(video.shape[0]):
        trace[tt] = np.mean(video[tt, :])


@jit("void(u2[:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fast_median_u2(video, bgtrace):
    for tt in prange(video.shape[0]):
        bgtrace[tt] = np.median(video[tt, :])


@jit("void(f4[:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fast_mean_f4(video, trace):
    for tt in prange(video.shape[0]):
        trace[tt] = np.mean(video[tt, :])


@jit("void(f4[:,:],f4[:])",nopython=True,parallel=True,cache=True,fastmath=True)
def fast_median_f4(video, bgtrace):
    for tt in prange(video.shape[0]):
        bgtrace[tt] = np.median(video[tt, :])


# import line_profiler
# # profile = line_profiler.LineProfiler()
# @profile
def traces_bgtraces_from_masks_numba_neighbors(video, masks):
    # Generate background traces for each neuron from ground truth masks
    (T, Lx, Ly) = video.shape
    (ncells, Lxm, Lym) = masks.shape

    (list_neighbors, comx, comy, area, r_bg) = find_neighbors(masks)
    [xx, yy] = np.meshgrid(np.arange(Ly), np.arange(Lx))
    xx = xx.astype('uint16')
    yy = yy.astype('uint16')

    # try:
    video = video.reshape((T, Lxm*Lym))
    # except:
    #     # video = video[:, :Lxm, :Lym].reshape((T, Lxm*Lym))
    #     raise ValueError('Shapes of video and masks do not match')

    traces = np.zeros((T, ncells), dtype='float32')
    bgtraces = np.zeros((T, ncells), dtype='float32')
    outtraces = np.zeros((T, ncells), dtype='float32')

    if video.dtype == 'uint16':
        fast_mean = fast_mean_u2
        fast_median = fast_median_u2
    elif video.dtype == 'float32':
        fast_mean = fast_mean_f4
        fast_median = fast_median_f4
    for nn in range(ncells):
        mask = masks[nn]
        r_bg_0 = max(r_bg, np.sqrt(area[nn]/np.pi) * 2.5)
        circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_0 ** 2
        # bgtraces[:, nn] = np.median(video[:, circleout.ravel()], 1)
        # traces[:, nn] = video[:, masks[nn].ravel()].mean(1)
        fast_mean(video[:, mask.ravel()], traces[:, nn])
        fast_median(video[:, circleout.ravel()], bgtraces[:, nn])

        fgmask = masks[list_neighbors[nn]].sum(0) > 0
        r_bg_large = r_bg * 0.8
        circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_large ** 2
        # bgmask = np.logical_and(circleout - fgmask, circleout) 
        bgmask = np.logical_and(circleout,np.logical_not(fgmask))
        bgweight = bgmask.sum()
        # if all pixels in circleout belongs to one of the neuron, than enlarge circleout to include more pixels.
        # if bgweight < area/2:
        while bgweight < area[nn]/2:
            r_bg_large = r_bg_large+1
            circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_large ** 2
            # bgmask = np.logical_and(circleout - fgmask, circleout) 
            bgmask = np.logical_and(circleout,np.logical_not(fgmask))
            bgweight = bgmask.sum()
        fast_mean(video[:, bgmask.ravel()], outtraces[:, nn])
    # fasttraces(video, masks, traces, bgtraces, xx, yy, r_bg)

    return traces, bgtraces, outtraces, list_neighbors


# def bgtraces_from_masks(video, masks):
#     # Generate background traces for each neuron from ground truth masks
#     (T, Lx, Ly) = video.shape
#     (ncells, Lxm, Lym) = masks.shape

#     try:
#         video = video.reshape((T, Lxm*Lym))
#     except:
#         # video = video[:, :Lxm, :Lym].reshape((T, Lxm*Lym))
#         raise ValueError('Shapes of video and masks do not match')

#     bgtraces = np.zeros((T, ncells), dtype='float32')
#     [xx, yy] = np.meshgrid(np.arange(Ly), np.arange(Lx))
#     r_bg = np.sqrt(masks.sum(-1).sum(-1).mean()/np.pi)*2.5
#     for nn in range(ncells):
#         mask = masks[nn]
#         [xxs, yys] = mask.nonzero()
#         comx = xxs.mean()
#         comy = yys.mean()
#         circleout = (yy-comx) ** 2 + (xx-comy) ** 2 < r_bg ** 2
#         bgtraces[:, nn] = np.median(video[:, circleout.ravel()], 1)

#     return bgtraces


def find_neighbors(FinalMasks):
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
        r = np.sqrt((comx[i]-comx) **2 + (comy[i]-comy) **2)
        neighbors = np.concatenate([np.array([i]), np.logical_and(r > 0, r < r_bg).nonzero()[0]])
        list_neighbors.append(neighbors)

    return list_neighbors, comx, comy, area, r_bg


