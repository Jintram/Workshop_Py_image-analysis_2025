"""
Bacterial segmentation pipeline, which uses the functions defined in 
`bacterial_seg_functions.py` to segment each frame in time-lapse tiff file,
and store resulting segmentation masks as npz files in output folder.
"""

################################################################################
# %% Import libraries

import sys, os
import yaml
import time, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from skimage.util import invert

import seaborn as sns
import tifffile as tiff
import numpy as np

################################################################################
# %% Custom imports

from . import bacterial_seg_functions as sf

################################################################################
# %% Plotting function

def plot_sidebyside(img_inv_crop, segmask_oneframe, frame_idx):

    # set font to arial 8
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8

    # create figure
    fig, axs = plt.subplots(1, 2, figsize=(10/2.54, 5/2.54))

    axs[0].imshow(img_inv_crop, cmap='gray')
    axs[0].set_title(f"Frame {frame_idx}")
    axs[1].imshow(segmask_oneframe, cmap=sf.cmap_random(200))
    axs[1].set_title(f"Segmentation")
    
    return fig

################################################################################
# %% Function to run the whole pipeline

def run_seg_pipeline(config, make_plots = True):
    """
    Runs the whole pipeline.
     
    Config is a dict with the following parameters:
    - path_input: path to the input time lapse tiff file (frame * x * y)
    - path_output: path to the output folder
    - frame_range: tuple with the range of frames to process (start, end)
    - disksize_range:  disk radius for global mask, should be >radius bacteria (e.g. 20)
    - sigma_smooth: smoothing sigma for the LoG (set e.g. to 3)
    - disksize_crossing: disk radius for neighborhood that defines zero-crossing (e.g. 1)
    - min_distance: minimum distance for local peak finder to identify bacteria,
        set to <bacterial radius (e.g. 5)
    - sigma_seed: gaussian blur applied to distance mask of preliminary mask to
        identify unique bacteria (e.g. 3)
    - distance_threshold: pixels with distance-to-background values >distance_threshold
        are kept to identify individual bacteria. Should be <expected minimum width
        of bacteria. (E.g. 3)

    Example:
    config = {
        "path_input": "/path/to/input/folder",
        "path_output": "/path/to/output/folder",
        "filename_data": "input.tiff",
        "frame_range": (0, 100),
        "disksize_range": 20,
        "sigma_smooth": 3,
        "disksize_crossing": 1,
        "min_distance": 5,
        "sigma_seed": 3,
        "distance_threshold": 3
    } 
    """

    # set up output folder (take name of input subfolder as identifier)
    basename = Path(config['filename_data']).stem
    output_subfolder = os.path.join(config['path_output'], basename)
    # create output sub folders
    os.makedirs(os.path.join(output_subfolder, "seg"), exist_ok=True)
    os.makedirs(os.path.join(output_subfolder, "plots"), exist_ok=True)    
    # Write a log file to the output folder    
    config['scriptname'] = os.path.basename(__file__) # Add this script's name 
    yaml.dump(config, open(os.path.join(output_subfolder, 'config_dump.yaml'), 'w'))
    
    # Load data
    imgs_ecoli = tiff.imread(os.path.join(config['path_input'],config['filename_data']))
    
    # loop over frames
    total_time = 0
    frames_to_process = range(config['frame_range'][0], config['frame_range'][1])
    for idx, frame_idx in enumerate(frames_to_process):
        # frame_idx = 254
    
        # update for user
        print(f"Processing frame {frame_idx}")
        # (gimmick) get current time
        current_time = time.time()
        
        # skip if input image is empty
        if len(np.unique(imgs_ecoli[frame_idx,:,:].ravel())) < 2:
            print("SKIPPING, INPUT IMAGE SEEMS TO BE EMPTY.")
            continue
    
        # now segment a frame
        segmask_oneframe, img_inv_crop = sf.seg_bacterium(
            input_img = invert(imgs_ecoli[frame_idx,:,:]), 
            sigma_smooth=config['sigma_smooth'], 
            disksize_crossing=config['disksize_crossing'], 
            disksize_range=config['disksize_range'], 
            min_distance=config['min_distance'], 
            sigma_seed=config['sigma_seed'], 
            distance_threshold=config['distance_threshold']
        )
        
        # create a side-by-side plot if make_plots == True
        if make_plots:
            fig = plot_sidebyside(img_inv_crop, segmask_oneframe, frame_idx)
            fig.savefig(
                os.path.join(
                    output_subfolder, "plots/", 
                    f"png_segmentation_frame{frame_idx:04d}.png"
                    ), 
                dpi=600
                )
            plt.close(fig)
    
        # save the segmentation to npz in appropriate output subfolder
        output_path = os.path.join(
            output_subfolder, 'seg/',  
            f"segmentation_frame_{frame_idx:04d}.npz"
            )
        np.savez(
            output_path, 
            segmask=segmask_oneframe,
            img_inv_crop=img_inv_crop
            )
        
        # (gimmick) update user about expected time remaining
        time_delta = time.time() - current_time
        total_time += time_delta
        remaining_time = (total_time/(idx+1))*(len(frames_to_process)-idx-1)
        remaining_time_str = datetime.timedelta(seconds=int(remaining_time))
        print(f"Time for this frame: {datetime.timedelta(seconds=time_delta)}.\nEstimated time remaining: {remaining_time_str}.")
        
# %%
