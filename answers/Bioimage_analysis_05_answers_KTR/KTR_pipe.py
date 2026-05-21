

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
# %%

def cmap_random(N):
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
    
    mask_rings = dilation(seg_mask, disk(width))
    mask_rings[seg_mask>0] = 0
    
    return mask_rings
    

def get_mean_intensity(img, mask):
    """Calculate means in areas corresponding to labeled mask in image img."""
    
    # Calculate totals per index
    sums   = np.bincount(mask.ravel(), weights=img.ravel())
    # Calculate pixel coverage per index
    counts = np.bincount(mask.ravel())
    
    # Calculate the means
    return (sums/counts)[1:]

################################################################################
# %%

if __name__ == "__main__":

    # Input arguments
    img_path_KTR = sys.argv[1]
    channel_nuc = sys.argv[2]
    channel_cyt = sys.argv[3]

    # Load data
    # img_path_KTR = '/Users/m.wehrens/Data_notbacked/2025_Py-Image-workshop_KTR-example-data/raw/Composite_KTR.tif'
    # channel_nuc = 0; channel_cyt = 2
    KTR_data = tiff.imread(img_path_KTR)

    # Initialize empty list to store dataframes
    df_KTR_list = [None]*KTR_data.shape[0]
    # Loop over the frames
    for fr_idx in range(0, KTR_data.shape[0]):
        # fr_idx=0
        
        print(f"Working on frame {fr_idx} / {KTR_data.shape[0]}")
        
        # get nuclei and KTR intensity images
        img_nuc = KTR_data[fr_idx, channel_nuc, 0:200, 0:200]
        img_KTR = KTR_data[fr_idx, channel_cyt, 0:200, 0:200]
            # plt.imshow(img_nuc)
            # plt.imshow(img_KTR)
            
        # Calculate the mask
        mask_nuclei       = seg_nuclei(img_nuc)
        mask_nuclei_split = separate_nuclei(mask_nuclei, split_distance=5)

        # Calculate averages
        KTR_means_nucl = get_mean_intensity(img_KTR, mask_nuclei_split)
        KTR_means_cyto = get_mean_intensity(img_KTR, get_rings(mask_nuclei_split))
            # plt.imshow(get_rings(mask_nuclei_split), cmap=cmap_random(200))

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

# Save the data in a new directory "analysis" in the parent directory
# (This assumes the raw data is stored in its own subdirectory in sample folder)
OUTPUT_PATH = Path(os.path.dirname(img_path_KTR)).parent / "analysis"
os.makedirs(OUTPUT_PATH, exist_ok=True)

output_path = os.path.join(OUTPUT_PATH, f"KTR_ratios_ch{channel_cyt}.csv")
df_KTR.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")

# save in the data directory


