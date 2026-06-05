""" Analyze KTR data to get C/N ratios

Segments nuclei in KTR data, generates cytoplasmic ROIs, 
calculates C/N ratios, and saves the data in a csv file.

Exports a csv file with columns "frame", "KTR_ratio", "int_nucleus", "int_cytoplasm",
where KTR_ratio is the C/N ratio of average cytoplasmic intensity (C) and 
average nuclear intensity (N).

Example of how to call script:
python KTR_pipe.py /path/to/Composite_KTR.tif 0 2

With:
- Argument 1: path to the raw KTR data (tif file)
- Argument 2: channel number for nuclear marker (counting starts at 0)
- Argument 3: channel number for KTR sensor (counting starts at 0)

"""

################################################################################
# %% Import libraries

import sys, os
import yaml
import time, datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns
import tifffile as tiff
import numpy as np
import pandas as pd

import skimage as sk
from skimage.util import invert

from scipy import stats
from scipy import ndimage

from skimage.measure import label, regionprops
from skimage.feature import peak_local_max

from skimage.morphology import dilation, disk


################################################################################
# %% helper functions for segmentation and analysis

def cmap_random(N):
    """ generate a cmap with random colors (first one black) """
    cmap = np.zeros((N+1, 3))
    for i in range(1, N+1):
        cmap[i] = np.random.rand(3)*0.7 + 0.3
    return ListedColormap(cmap)

def seg_nuclei(input_img):
    """
    Segment nuclei in an image using triangle thresholding.
    """
    
    # Apply triangle threshold method
    thresh = sk.filters.threshold_triangle(input_img)
    mask = input_img > thresh
    
    # Remove small objects and label
    mask_filtered = sk.morphology.remove_small_objects(mask, min_size=50)
    result = sk.measure.label(mask_filtered)
    
    return result


def separate_nuclei(seg_mask, split_distance=5, showplot=False):
    """
    Apply watershed function, based on distance transform.
    See also https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_watershed.html
    """
    
    # distance transform
    mask_distance = ndimage.distance_transform_edt(seg_mask)
    
    # blur distance transform to remove noise
    mask_distance_blurred = sk.filters.gaussian(mask_distance, sigma=split_distance/2)
    
    # find the seeds using local maxima
    local_minima = peak_local_max(mask_distance_blurred, 
                                  footprint=disk(split_distance),
                                  exclude_border=False)
    mask_seeds = np.zeros_like(mask_distance, dtype=int)
    mask_seeds[local_minima[:,0], local_minima[:,1]] = 1
    mask_seeds = label(mask_seeds)
    
    if showplot: 
        plt.figure()
        plt.imshow(mask_distance_blurred)
        plt.contour(seg_mask>0, levels=[0.5], colors="r")
        for x,y in local_minima:
            plt.plot(y,x, "ro")
        plt.show()

    # perform watershed
    mask_final = sk.segmentation.watershed(image=-seg_mask,
                                           markers=mask_seeds,
                                           mask=seg_mask>0)
            
    if showplot:
        plt.figure()
        plt.imshow(mask_final, cmap=cmap_random(200))
        plt.show()
        
    return mask_final
        

def get_rings(seg_mask, width=2):
    """ Based on a mask, get areas exactly adjacent to the masks (aka 'rings')
    
    These areas should correspond to cytoplasm. """ 
    
    # dilate to get expanded area
    mask_rings = dilation(seg_mask, disk(width))
    # then remove original to get rings
    mask_rings[seg_mask>0] = 0
    
    return mask_rings
    
def get_rings_margin(seg_mask, width=2, margin=1):
    """ Based on mask, get areas directly adjacent to the masks after margin 
    
    These areas should correspond to cytoplasm.
    """
    
    # dilate to get expanded area (plus margin)
    mask_rings = dilation(seg_mask, disk(width+margin))
    # remove original + margin to get ring around nucleus, with margin
    mask_rings[dilation(seg_mask, disk(margin))>0] = 0
    
    return mask_rings
        

def correct_bg(img, export_path=None):
    """
    Apply background correction to image.
     
    Works by determining the mode, and subtracting it.
    Resulting negative values are set to zero.
    
    If export_path is given, export picture.
    """
    # img = img_KTR; export_path = "/Users/m.wehrens/Desktop/test.png"
    
    # determine background level and mask
    the_mode = np.bincount(img.ravel()).argmax()
    mask_bg  = img<the_mode
    
    # subtract it 
    img_corr = img - the_mode
    # set would-be negative values to 0
    img_corr[mask_bg] = 0 
    
    if export_path is not None:
        
        fig, axs = plt.subplots(1,3, figsize=(15/2.54,5/2.54))
        plt.rcParams.update({"font.family": "Arial", "font.size": 7})
        
        # Show image, and image with identified background pixels
        axs[0].imshow(img)
        axs[1].imshow(img)
        axs[1].imshow(mask_bg, alpha=mask_bg*1.0)

        # show it in the histogram
        axs[2].hist(img.ravel(), bins=32, color='grey')
        axs[2].axvline(the_mode, color='red')
        axs[2].set_xlabel("Intensity")
        axs[2].set_ylabel("Counts")
        
        plt.tight_layout()
        fig.savefig(export_path, dpi=600)
        plt.close()
        
    return img_corr
        

def get_mean_intensity(img, mask):
    """Calculate means in areas corresponding to labeled mask in image img."""
    
    # Calculate totals per index
    sums   = np.bincount(mask.ravel(), weights=img.ravel())
    # Calculate pixel coverage per index
    counts = np.bincount(mask.ravel())
        
    # Calculate the means
    return (sums/counts)[1:]

################################################################################
# %% Code executed when script is called

if __name__ == "__main__":

    # Input arguments
    img_path_KTR = sys.argv[1]
    channel_nuc = int(sys.argv[2])
    channel_ktr = int(sys.argv[3])

    # Set up an output directory named "analysis" in the parent directory
    # (This assumes the raw data is stored in its own subdirectory in sample folder)
    # e.g.
    # ../../projectX/data/sampleXYZ_202606/raw/
    # ../../projectX/data/sampleXYZ_202606/analysis/
    output_dir = Path(os.path.dirname(img_path_KTR)).parent / "analysis"
    os.makedirs(output_dir, exist_ok=True)
    path_bg_fig = os.path.join(output_dir, f"background_correction_{channel_ktr}.png")

    # Load data
    # img_path_KTR = '/Users/m.wehrens/Data_notbacked/2025_Py-Image-workshop_KTR-example-data/raw/Composite_KTR.tif'
    # channel_nuc = 0; channel_ktr = 2
    KTR_data = tiff.imread(img_path_KTR)

    # Initialize empty list to store dataframes
    df_KTR_list = [None]*KTR_data.shape[0]
    # Loop over the frames
    for fr_idx in range(0, KTR_data.shape[0]):
        # fr_idx=0
        
        print(f"Working on frame {fr_idx} / {KTR_data.shape[0]}")
        
        # get nuclei and KTR intensity images
        img_nuc = KTR_data[fr_idx, channel_nuc, :, :]
        img_KTR = KTR_data[fr_idx, channel_ktr, :, :]
            # plt.imshow(img_nuc)
            # plt.imshow(img_KTR)
            
        # correct the background
        if fr_idx == 0:
            # if first frame, also export image to check whether
            # background correction was appropriate
            img_KTR = correct_bg(img_KTR, export_path=path_bg_fig)
        else:
            img_KTR = correct_bg(img_KTR)
            
        # Calculate the mask
        mask_nuclei       = seg_nuclei(img_nuc)
        mask_nuclei_split = separate_nuclei(mask_nuclei, split_distance=5)

        # Calculate averages
        KTR_means_nucl = get_mean_intensity(img_KTR, mask_nuclei_split)
        KTR_means_cyto = get_mean_intensity(img_KTR, get_rings_margin(mask_nuclei_split))
            # plt.imshow(get_rings(mask_nuclei_split), cmap=cmap_random(200))
            # CM = cmap_random(200); plt.imshow(mask_nuclei_split, cmap=CM); plt.imshow(get_rings_margin(mask_nuclei_split), cmap=CM, alpha=(get_rings_margin(mask_nuclei_split)>1)*1.0)

        # Get ratio
        KTR_ratios = KTR_means_cyto/KTR_means_nucl
        
        # Store the data
        df_KTR_list[fr_idx] = pd.DataFrame({
            "frame": fr_idx, 
            "KTR_ratio": KTR_ratios,
            "int_nucleus": KTR_means_nucl,
            "int_cytoplasm": KTR_means_cyto
        })

    # Now merge the dataframes
    df_KTR = pd.concat(df_KTR_list, ignore_index=True)
   
    # Save data to csv
    output_path = os.path.join(output_dir, f"KTR_ratios_ch{channel_ktr}.csv")
    df_KTR.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")



# %%

if __name__ == "main":

    """
    python KTR_pipe.py /Users/m.wehrens/Data_notbacked/2025_Py-Image-workshop_KTR-example-data/raw/Composite_KTR.tif 0 2
    """