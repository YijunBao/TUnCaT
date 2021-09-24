![NeuroToolbox logo](readme/neurotoolbox-logo.svg)

# TUnCaT
Temporal Unmixing of Calcium Traces (TUnCaT) is an automatic algorithm to decontaminate false transients from temporal traces generated from fluorescent calcium imaging videos. TUnCaT removed false transients caused by large-scale background fluctuation using background subtraction; TUnCaT removed false transients caused by neighboring neurons, axons, and dendrites using nonnegative matrix factorization (NMF).

Copyright (C) 2021 Duke University NeuroToolbox

This repo is an archived version of TUnCaT (v0.1). It also contains the results for our paper and the code to generate those results. This repo will not be updated. Please visit the [newer version](https://github.com/YijunBao/TUnCaT) if you are not interested in reproducing our paper results. 

- [TUnCaT](#tuncat)
- [Installation on Windows or Linux](#installation-on-windows-or-linux)
  - [From official Python website](#from-official-python-website)
  - [From Anaconda](#from-anaconda)
- [Demo](#demo)
- [Input, Output, and Intermediate Files](#input-output-and-intermediate-files)
  - [Input files](#input-files)
  - [Intermediate and Output files](#intermediate-and-output-files)
- [Use your own data](#use-your-own-data)
  - [Use your own videos](#use-your-own-videos)
  - [Use your own neuron masks](#use-your-own-neuron-masks)
  - [Set the unmixing parameters](#set-the-unmixing-parameters)
- [Citation](#citation)
- [Licensing and Copyright](#licensing-and-copyright)
- [Sponsors](#sponsors)


# Installation on Windows or Linux 
## From official Python website
* In Windows, install Python from the [official website](https://www.python.org/downloads/). 
* In Linux, Python is often already installed. You can also install the lastest version by typing the following commands
```sh
sudo apt-get install python3
sudo alias python = python3
```
* Launch Windows or Linux terminal, and install the required packages: numpy, scipy, h5py, scikit-learn, numba by typing the following commands.Because numpy and scipy are dependencies of scikit-learn, only the remaining three packages needs manual installation.
```bat
python -m pip install h5py
python -m pip install scikit-learn
python -m pip install numba
```
To run the optional figure plotting section in the demo, matplotlib should be installed
```bat
python -m pip install matplotlib
```

## From Anaconda
* Install Python [Anaconda](https://www.anaconda.com/)
* Launch Anaconda prompt, and install the required packages: numpy, scipy, h5py, scikit-learn, numba. Because numpy and scipy are preinstalled in Anaconda, only the remaining three packages needs manual installation.
```bat
conda install h5py
conda install scikit-learn
conda install numba
```
* Please be aware that the speed of TUnCaT in the Anaconda installation may be significantly slower than the speed in the offical python installation in Windows.


# Demo
We provided a demo for all users to get familiar with TUnCaT. We provided a one-photon imaging video `c28_163_244.h5` as well as its manually labeled neurons `FinalMasks_c28_163_244.mat`. The demo will calculate the raw traces and background traces of all neurons, calculate the unmixed traces using TUnCaT, and export them to the folder `unmixed_traces`. The input, output, and intermediate files will be explained in [Input, Output, and Intermediate Files](#input-output-and-intermediate-files). 

To run the demo python script, launch system terminal or Anaconda prompt, and type the following script 
```bat
cd {TUnCaT_root_path}
python demo_TUnCaT.py
```
You can also run the demo `demo_TUnCaT.ipynb` interactively using Jupyter Notebook. This notebook contains the expected figure of the background-subtracted trace and the unmixed trace of the first neuron. The saved processing time is recorded in a laptop with an AMD Ryzen 5 3500U quad-core CPU.


# Input, Output, and Intermediate Files
By default, all the input, output, and intermediate files are saved under the `TUnCaT` folder. 

## Input files
* A .h5 file `{name}.h5` contains the input video. This is a 3D dataset with shape = (T, Lx, Ly), where T is the number of frames, and (Lx, Ly) is the lateral dimension of each frame. The demo videos have (T, Lx, Ly) = (6000, 50, 50). 
* A .mat file `FinalMasks_{name}.mat` contains the nueron masks of the video. This is a 3D array with shape = (Ly, Lx, n) in MATLAB, where Ly and Lx should match the lateral dimensions of the video, and n is the number of neurons. The demo masks have (Ly, Lx, n) = (50, 50, 45). 
* Important notice: The default dimension order for multi-dimensional array is reversed in MATLAB and python. When you save a dataset with shape = (L1, L2, L3) in MATLAB to an .h5 file or a .mat file with version 7.3 or newer (requiring h5py.File to load in python workspace), and then load it in python, the shape will become (L3, L2, L1). However, if you save the dataset as a .mat file with version 7 or earlier (requiring scipy.loadmat to load in python workspace), the dimensions will preserve and still be (L1, L2, L3). In this document, we will use the python default order to describe the datasets in python workspace or saved in .h5, and use the MATLAB default order to describe the datasets saved in .mat. Sometimes you need to transpose the dimensions to make them consistent. In python, you can transpose the dimensions using `Masks = Masks.transpose((2,1,0))`. In MATLAB, you can transpose the dimensions using `Masks = permute(Masks,[3,2,1])`.

## Intermediate and Output files
After running TUnCaT on the demo video, the intermediate and output files will be under a new folder `unmixed_traces`. 
* Intermediate file: `unmixed_traces/raw/{name}.mat` stores the raw neuron traces (`traces`) and the background traces (`bgtraces`). Each trace variable is 2D matrix with shape = (T, n), where T is the number of frames, and n is the number of neurons. 
* Output file: `unmixed_traces/alpha={}/{name}.mat` or `unmixed_traces/list_alpha={}/{name}.mat` stores the unmixed traces of the neurons (`traces_nmfdemix`), as well as other quantities characterizing the NMF unmixing process of each neuron, including the mixing matrix (`list_mixout`), final alpha (`list_alpha_final`), reconstruction residual (`list_MSE`), and number of iterations (`list_n_iter`). See the function descriptions for the detailed meanings of these quantities.


# Use your own data
Of course, you can modify the demo scripts to process other videos. You need to set the folders of the videos and neuron masks, and change some parameters in the python scripts to correspond to your videos. 

## Use your own videos
* Set the folder and file names of your video correctly. You need to change the variables `dir_video` and `list_Exp_ID`. The variable `dir_video` is the folder containing the input videos. For example, if your videos are stored in `C:/Users/{username}/Documents/GitHub/TUnCaT_paper_reproduction/TUnCaT/data`, set `dir_video = 'C:/Users/{username}/Documents/GitHub/TUnCaT_paper_reproduction/TUnCaT/data'`. You can also use relative path, such as `dir_video = './data'`. The variable `list_Exp_ID` is the list of the file names (without extension) of the input videos (e.g., `list_Exp_ID = ['c28_163_244']` in the demo referes to the input file `{TUnCaT_root_path}/data/c28_163_244.h5`). 
* Currently, we only support .h5 files as the input video, so you need to convert the format of your data to .h5. You can save the video in a dataset with any name, but don't save the video under any group. The video should have a shape of (T, Lx, Ly), where T is the number of frames, and (Lx, Ly) is the lateral dimension of each frame. The support to more video formats will come soon. When doing format conversion, make sure the dimension is in the correct order. For example, if you save save the .h5 files from MATLAB, the shape of the dataset should be (Ly, Lx, T) in MATLAB. See [Input files](#input-files) for more explanation of the dimension order.

## Use your own neuron masks
* Set the folder and file names of your mask files correctly. You need to change variable `dir_mask` to the folder containing the mask files. 
* Currently, we only support .mat files as the neuron masks, so you need to convert the format of your neuron masks to .mat, and save the neuron masks in dataset 'FinalMasks'. The name of the masks file should be `FinalMasks_{Exp_ID}.mat`, where the `{Exp_ID}` is the name of the corresponding video. The neuron masks should be saved as a 3D array named `FinalMasks`, and the dimension should be (Ly, Lx, n) in MATLAB, where Ly and Lx are the same as the lateral dimension of the video, and n is the number of neurons.
* The masks created by some manual labeling software may contain an empty (all zero) image as the first frame. You need to remove the empty frame before saving them.

## Set the unmixing parameters
* The list of video names (e.g., `list_Exp_ID = ['c28_163_244']`)
* The folder of the video (e.g., `dir_video='./data'`);
* The folder of the neuron masks (e.g., `dir_masks='./data'`);
* The folder of the output traces (e.g., `dir_traces='./data/unmixed_traces'`);
* `list_alpha` is the list of tested alpha;
* `multi_alpha` determines the option to deal with multiple elements in `list_alpha`. False means the largest element providing non-trivial output traces will be used, which can be differnt for different neurons. True means each element will be tested and saved independently. These options are equivalent when there is only one element in `list_alpha`;
* `use_direction` indicates whether a direction requirement is applied to the output traces;
* `Qclip` is clipping quantile. Traces lower than this quantile are clipped to this quantile value;
* `th_pertmin` is maximum pertentage of unmixed traces equaling to the trace minimum;
* `epsilon` is the minimum value of the input traces after scaling and shifting;
* `nbin` is the temporal downsampling ratio;
* `bin_option` determines the temporal downsampling option. It can be 'downsample', 'sum', or 'mean'. It is not used when nbin == 1;
* `flexible_alpha` determines whether a flexible alpha strategy is used when the smallest alpha in "list_alpha" already caused over-regularization.
* Among these parameters, we think most parameters do not need to be changed, but an optimized `list_alpha` can improve the unmixing accuracy. In our paper, we optimized the alpha using cross-validation, but it requires manual labeling of many traces, which is very time consuming. Instead, you can start with `list_alpha = [1]`, because most of our optimized alpha are close to 1. 


# Citation 
If you use any part of this software in your work, please cite Bao et al. 2021.


# Licensing and Copyright
TUnCaT is released under [the GNU License, Version 2.0](LICENSE).


# Sponsors
<img src="readme/NSFBRAIN.png" height="100"/><img src="readme/BRF.png" height="100"/><img src="readme/Beckmanlogo.png" height="100"/>
<br>
<img src="readme/valleelogo.png" height="100"/><img src="readme/dibslogo.png" height="100"/><img src="readme/sloan_logo_new.jpg" height="100"/>
