import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory


# import line_profiler
# # profile = line_profiler.LineProfiler()
# @profile
def mean_median_out_trace(nn, shm_video, video_shape, video_dtype, \
    shm_masks, shm_xx, shm_yy, r_bg, comx, comy, area, neighbors):
    # shm_video = SharedMemory(name = video_name)
    (T, Lx, Ly) = video_shape
    n = comx.size
    video = np.ndarray((T, Lx*Ly), buffer=shm_video.buf, dtype = video_dtype)
    FinalMasks = np.ndarray((n, Lx, Ly), buffer=shm_masks.buf, dtype = 'bool')
    mask = FinalMasks[nn]
    # mask = np.ndarray((Lx,Ly), buffer=shm_masks.buf[nn*Lx*Ly:(nn+1)*Lx*Ly], dtype = 'bool')
    xx = np.ndarray((Lx,Ly), buffer=shm_xx.buf, dtype='uint16')
    yy = np.ndarray((Lx,Ly), buffer=shm_yy.buf, dtype='uint16')

    # mask = masks[nn]
    # [xxs, yys] = mask.nonzero()
    # comx = xxs.mean()
    # comy = yys.mean()
    # area = xxs.size
    r_bg_0 = max(r_bg, np.sqrt(area/np.pi) * 2.5)
    circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_0 ** 2
    bgtrace = np.median(video[:, circleout.ravel()], 1)
    # temp = video[:, circleout.ravel()]
    # bgtrace = np.median(temp, 1)
    # bgtrace = np.quantile(video[:, circleout.ravel()], 0.25, 1)
    trace = video[:, mask.ravel()].mean(1)
    # print(nn, circleout.sum(), bgtrace.max())

    fgmask = FinalMasks[neighbors].sum(0) > 0
    r_bg_large = r_bg * 0.8
    circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_large ** 2
    # bgmask = np.logical_and(circleout - fgmask, circleout) 
    bgmask = np.logical_and(circleout,np.logical_not(fgmask))
    bgweight = bgmask.sum()
    # if all pixels in circleout belongs to one of the neuron, than enlarge circleout to include more pixels.
    # if bgweight < area/2:
    while bgweight < area/2: # == 0: # 
        r_bg_large = r_bg_large+1
        circleout = (yy-comx[nn]) ** 2 + (xx-comy[nn]) ** 2 < r_bg_large ** 2
        # bgmask = np.logical_and(circleout - fgmask, circleout) 
        bgmask = np.logical_and(circleout,np.logical_not(fgmask))
        bgweight = bgmask.sum()
    outtrace = video[:, bgmask.ravel()].mean(1)
    # temp2 = video[:, bgmask.ravel()]
    # outtrace = temp2.mean(1)
    # print(nn,bgmask.sum())

    return trace, bgtrace, outtrace


def traces_bgtraces_from_masks_neighbors(shm_video, video_dtype, video_shape, \
        shm_masks, masks_shape, FinalMasks):
    # (T, Lx, Ly) = video_shape
    (ncells, Lx, Ly) = masks_shape
    (list_neighbors, comx, comy, area, r_bg) = find_neighbors(FinalMasks)

    [xx, yy] = np.meshgrid(np.arange(Ly), np.arange(Lx))
    xx = xx.astype('uint16')
    shm_xx = SharedMemory(create=True, size=xx.nbytes)
    xx_temp = np.frombuffer(shm_xx.buf, dtype='uint16')
    xx_temp[:] = xx.ravel()
    yy = yy.astype('uint16')
    shm_yy = SharedMemory(create=True, size=yy.nbytes)
    yy_temp = np.frombuffer(shm_yy.buf, dtype='uint16')
    yy_temp[:] = yy.ravel()

    # mean_median_trace(0, video_name, video_shape, video_dtype, \
    #     masks_name, masks_shape, xx_name, yy_name, xx.dtype, r_bg)
    
    # results = []
    # for nn in range(ncells):
    #     results.append(mean_median_out_trace(nn, shm_video, video_shape, video_dtype, \
    #         shm_masks, shm_xx, shm_yy, r_bg, comx, comy, area[nn], list_neighbors[nn]))

    p = mp.Pool(mp.cpu_count())
    results = p.starmap(mean_median_out_trace, [(nn, shm_video, video_shape, video_dtype, \
        shm_masks, shm_xx, shm_yy, r_bg, comx, comy, area[nn], list_neighbors[nn]) for nn in range(ncells)], chunksize=1)
    p.close()

    traces = np.vstack([x[0] for x in results]).T
    bgtraces = np.vstack([x[1] for x in results]).T
    outtraces = np.array([x[2] for x in results]).T
    # print(bgtraces.max(0))

    return traces, bgtraces, outtraces, list_neighbors


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

