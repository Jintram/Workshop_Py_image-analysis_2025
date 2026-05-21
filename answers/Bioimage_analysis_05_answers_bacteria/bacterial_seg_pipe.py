"""

!!ADD A SUMMARY HERE!!



"""

################################################################################
# %% Import libraries

import sys, os
import yaml
import time, datetime

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns
import tifffile as tiff
import numpy as np

import skimage as sk
from skimage.util import invert

from scipy import stats
from scipy import ndimage

from skimage.measure import label, regionprops

################################################################################
# %% Segmentation functions

    
def plot_sidebyside(img_inv_crop, segmask_oneframe, frame_idx):

    # set font to arial 8
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8

    # create figure
    fig, axs = plt.subplots(1, 2, figsize=(10/2.54, 5/2.54))

    axs[0].imshow(img_inv_crop, cmap='gray')
    axs[0].set_title(f"Frame {frame_idx}")
    axs[1].imshow(segmask_oneframe, cmap=cmap_random(200))
    axs[1].set_title(f"Segmentation")
    
    return fig

################################################################################
# %% Apply it

# This statements checks if the script is being run directly
if __name__ == "__main__":

    # read config file
    # config_file_path = "answers/Bioimage-analysis_05_answers_bacteria/config_bacteria_seg.yaml"
    config_file_path = sys.argv[1]
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # set up output folder
    basename = config['path_input'].split('/')[-1].split('.')[0]
    output_subfolder = os.path.join(config['path_output'], basename)
    # create output sub folder
    os.makedirs(output_subfolder, exist_ok=True)
    os.makedirs(os.path.join(output_subfolder, "plots"), exist_ok=True)    
    # write the configuration file to the subfolder
    yaml.dump(config, open(os.path.join(output_subfolder, 'config_dump.yaml'), 'w'))
    
    # Load data
    imgs_ecoli = tiff.imread(config['path_input'])
    
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
        segmask_oneframe, img_inv_crop = seg_bacterium(
            input_img = invert(imgs_ecoli[frame_idx,:,:]), 
            sigma_smooth=config['sigma_smooth'], 
            disksize_crossing=config['disksize_crossing'], 
            disksize_range=config['disksize_range'], 
            min_distance=config['min_distance'], 
            sigma_seed=config['sigma_seed'], 
            distance_threshold=config['distance_threshold']
        )
        
        # create a side-by-side plot if sys.argv[2]=="plot"
        if len(sys.argv) > 2:
            if sys.argv[2] == "plot":
                fig = plot_sidebyside(img_inv_crop, segmask_oneframe, frame_idx)
                fig.savefig(
                    os.path.join(
                        output_subfolder, "plots/", 
                        f"png_segmentation_frame{frame_idx:04d}.png"
                        ), 
                    dpi=600
                    )
    
        # save it to npz in output folder
        output_path = os.path.join(
            output_subfolder, 
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
