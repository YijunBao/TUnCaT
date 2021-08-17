# TUnCaT
Temporal Unmixing of Calcium Traces

**Important note**: The current version requires python 3.8 or newer. A version compatible with python 3.7 or older will come soon.  

* Required packages: numpy, scipy, h5py, scikit-learn, numba

# Demo 
* Script: `demo_nmfunmix.py`
* Parameter: 
  * Regularization parameter: `alpha` (`list_alpha` is a list of `alpha` to be tried)
  * Video location: `dir_video`
  * List of video names: `list_Exp_ID` (should be .h5 files)

* Inputs:
  * Video: `c28_163_244.h5`
  * GT neuron masks: `FinalMasks_c28_163_244.mat`

* Output folder: `unmixed_traces`
  * Raw traces: `unmixed_traces/raw/c28_163_244.mat`
  * Unmixed traces: `unmixed_traces/alpha= 1.000/c28_163_244.mat`