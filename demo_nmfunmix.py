import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import h5py

# if (sys.version_info.major+sys.version_info.minor/10)>=3.8
try:
    from multiprocessing.shared_memory import SharedMemory
    from traces_from_masks_numba_neighbors import traces_bgtraces_from_masks_numba_neighbors
    # from traces_from_masks_mp_share import bgtraces_from_masks, traces_from_masks, traces_bgtraces_from_masks
    # from traces_from_masks_mp_share import traces_bgtraces_from_masks
    from traces_from_masks_mp_shm_neighbors import traces_bgtraces_from_masks_neighbors
    # from traces_from_masks_nooverlap_mp_shm import traces_bgtraces_from_masks
    from use_nmfunmix_mp_diag_v1_shm_MSE_novideo import use_nmfunmix
    # from r_neuropil_stdmin import estimate_contamination_ratios
except:
    raise ImportError('No SharedMemory module. Please use Python version >=3.8, or use memory mapping instead of SharedMemory')


if __name__ == '__main__':
    # sys.argv = ['py', '0', 'Raw', '1', '1', '0', '0']
    # list_alpha = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30]  # 
    # list_alpha = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10] # 
    # list_alpha = [0.01, 0.02, 0.03, 0.05] # 
    list_alpha = [1] # 
    dir_video = '.'
    list_Exp_ID = ['c28_163_244']
    Table_time = np.zeros((len(list_Exp_ID), len(list_alpha)+1))
    # video_type = sys.argv[2]
    video_type = 'SNR' # 'Raw' # 
    # eid_select = int(sys.argv[1])

    Qclip = 0  # 0.08 # 
    nbin = 1 # int(sys.argv[3]) # 
    # if nbin == 1:
    bin_option = 'downsample'
        # addon = ''
    # else:
    #     bin_option = 'sum' # # sys.argv[1] # 'mean' # 
    #     if bin_option == 'mean':
    #         list_alpha = [x/10 for x in list_alpha]
    #         addon = '_'+bin_option +str(nbin)
    #     else:
    #         addon = '_'+bin_option +str(nbin)

    th_pertmin = 1 # float(sys.argv[4])
    epsilon = 0 # float(sys.argv[5])
    use_direction = False # bool(int(sys.argv[6]))
    # if th_pertmin < 1:
    #     addon += '_pertmin='+str(th_pertmin)
    # if epsilon > 0:
    #     addon += '_eps='+str(epsilon)
    # if use_direction:
    #     addon += '_range2'

    # Load video and FinalMasks
    # if video_type == 'SNR':
    #     varname = 'network_input' # 
    #     dir_video_SNR = os.path.join(dir_video, 'SNR video')
    # else:
    varname = 'mov' # 
    dir_video_SNR = dir_video
    dir_masks = dir_video # os.path.join(dir_video, 'GT Masks merge')
    dir_traces = os.path.join(dir_video, 'unmixed_traces_')
    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 
    dir_trace_raw = os.path.join(dir_traces, "raw")
    if not os.path.exists(dir_trace_raw):
        os.makedirs(dir_trace_raw)        
    
    for (ind_Exp, Exp_ID) in enumerate(list_Exp_ID):
        # if ind_Exp > eid_select:
        #     continue
        print(Exp_ID)
        start = time.time()
        filename_video = os.path.join(dir_video_SNR, Exp_ID + '.h5')
        file_video = h5py.File(filename_video, 'r')
        (T, Lx, Ly) = video_shape = file_video[varname].shape
        video_dtype = file_video[varname].dtype
        nbytes_video = int(video_dtype.itemsize * file_video[varname].size)
        shm_video = SharedMemory(create=True, size=nbytes_video)
        video = np.frombuffer(shm_video.buf, dtype = file_video[varname].dtype)
        video[:] = file_video[varname][()].ravel()
        video = video.reshape(file_video[varname].shape)
        # video_name = shm_video.name
        file_video.close()

        filename_masks = os.path.join(dir_masks, 'FinalMasks_' + Exp_ID + '.mat')
        try:
            file_masks = loadmat(filename_masks)
            Masks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
        except:
            file_masks = h5py.File(filename_masks, 'r')
            Masks = np.array(file_masks['FinalMasks']).astype('bool')
            file_masks.close()
        (ncells, Lxm, Lym) = masks_shape = Masks.shape
        shm_masks = SharedMemory(create=True, size=Masks.nbytes)
        FinalMasks = np.frombuffer(shm_masks.buf, dtype = 'bool')
        FinalMasks[:] = Masks.ravel()
        FinalMasks = FinalMasks.reshape(Masks.shape)
        # masks_sum = FinalMasks.astype('uint8').sum(0)
        # masks_name = shm_masks.name
        del Masks
        finish = time.time()
        print(finish - start)

        # Calculate traces and background traces
        start = time.time()
        # r_bg = np.sqrt(FinalMasks.sum(-1).sum(-1).mean()/np.pi)*2.5
        # [xx, yy] = np.meshgrid(np.arange(Ly), np.arange(Lx))
        # xx = xx.astype('uint16')
        # shm_xx = SharedMemory(create=True, size=xx.nbytes)
        # xx_temp = np.frombuffer(shm_xx.buf, dtype='uint16')
        # xx_temp[:] = xx.ravel()
        # yy = yy.astype('uint16')
        # shm_yy = SharedMemory(create=True, size=yy.nbytes)
        # yy_temp = np.frombuffer(shm_yy.buf, dtype='uint16')
        # yy_temp[:] = yy.ravel()

        if FinalMasks.sum()*T >= 7e7: # Use multiprocessing is faster for large videos
            (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_neighbors(shm_video, video_dtype, \
                video_shape, shm_masks, masks_shape, FinalMasks) # , masks_sum
        else: # Use numba is faster for small videos
            (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_numba_neighbors(
                video, FinalMasks) # , masks_sum

        # r = estimate_contamination_ratios(traces, bgtraces)
        # r_dict = estimate_contamination_ratios(traces, bgtraces)
        # r = r_dict['r']
        # bgtraces *= r
        finish = time.time()
        print('trace calculation time: {} s'.format(finish - start))
        Table_time[ind_Exp, -1] = finish-start

        # Save the raw traces into a ".mat" file under folder "dir_trace_raw".
        savemat(os.path.join(dir_trace_raw, Exp_ID+".mat"), {"traces": traces, "bgtraces": bgtraces})

        for (ind_alpha, alpha) in enumerate(list_alpha):
            print(Exp_ID, 'alpha =', alpha)
            # Do NMF unmixing
            start = time.time()
            traces_nmfdemix, list_mixout, list_MSE, list_final_alpha, list_n_iter = use_nmfunmix(traces, bgtraces, outtraces, \
                list_neighbors, [alpha], Qclip, th_pertmin, epsilon, use_direction, nbin, bin_option)
            # traces_nmfdemix = traces - bgtraces
            finish = time.time()
            print('unmixing time: {} s'.format(finish - start))
            Table_time[ind_Exp, ind_alpha] = finish-start

            # Save the unmixed traces into a ".mat" file under folder "dir_trace_unmix".
            dir_trace_unmix = os.path.join(dir_traces, "alpha={:6.3f}".format(alpha))
            if not os.path.exists(dir_trace_unmix):
                os.makedirs(dir_trace_unmix)        
            savemat(os.path.join(dir_trace_unmix, Exp_ID+".mat"), {"traces_nmfdemix": traces_nmfdemix,\
                "list_mixout":list_mixout, "list_MSE":list_MSE, "list_final_alpha":list_final_alpha, "list_n_iter":list_n_iter})

        # shm_xx.close()
        # shm_xx.unlink()
        # shm_yy.close()
        # shm_yy.unlink()
        shm_video.close()
        shm_video.unlink()
        shm_masks.close()
        shm_masks.unlink()

        savemat(os.path.join(dir_traces, "Table_time.mat"), {"Table_time": Table_time, 'list_alpha': list_alpha})

