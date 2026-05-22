"""
Read seg files from a specific location and plot their sizes, assuming the
filename ends with frame number, and has format <name>_0000.npz.
"""

# %% 

import glob
import numpy as np
import os
import pandas as pd
import seaborn as sns

################################################################################
# %% Helper functions

def load_segmask(filepath):
    """Load a segmentation mask"""
    data = np.load(filepath)
    return data["segmask"]

# get bacterial sizes
def calculate_sizes(segmask):
    """return sizes of labeled objects"""
    return np.bincount(segmask.flatten())[1:]  # skip background (0)

################################################################################
# %% Plotting function

def get_bacterial_sizes(current_folder):
    """Go over segmentation masks stored in a folder, and calculate 
    bacterial sizes. Segmentation masks should be labeled. Files
    in the folder should end with frame number, and have format <name>_0000.npz.

    Args:
        current_folder: path to folder with segmentation masks
    """

    # current_folder = "/Users/m.wehrens/Data_notbacked/2025_Py-Image-workshop_OUTPUT-examples/Filamentation/pos3crop_timelapse_switch890"

    # get all files
    file_list = glob.glob(current_folder + "/*.npz")

    # get all frame numbers
    frame_numbers = [int(filename.split("/")[-1].split(".")[0].split("_")[-1]) for filename in file_list]
    frames_sorted = np.sort(frame_numbers)

    # get file base name
    file_basename = "_".join(file_list[0].split("/")[-1].split(".")[0].split("_")[:-1])

    # set up dataframe
    df_sizes = pd.DataFrame(columns=["frame", "size"])

    # load file
    for frame_idx in frames_sorted:
        # frame_idx = 0
        filepath_segfile = os.path.join(
                    current_folder, f"{file_basename}_{frame_idx:04d}.npz"
                    )

        # load segfile & calculate sizes
        segmask = load_segmask(filepath_segfile)
        current_sizes = calculate_sizes(segmask)

        # add this info to dataframe
        df_sizes = pd.concat(
            [df_sizes, pd.DataFrame({"frame": frame_idx, "size": current_sizes})],
            ignore_index=True
        )
    
    return df_sizes


