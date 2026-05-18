

################################################################################
# %% Import libraries

import sys, os
import yaml

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

# create a cmap with 500 random somewhat bright colors, and starting with black
def cmap_random(N):
    cmap = np.zeros((N+1, 3))
    for i in range(1, N+1):
        cmap[i] = np.random.rand(3)*0.7 + 0.3
    return ListedColormap(cmap)

def _global_mask(input_img, disksize_range=20):
    """ Determine a global mask based on local intensity differences."""
    # input_img = invert(imgs_ecoli[10,:,:])
   
    # determine local difference ranges
    # - "img_delta" will contain the difference between the local maxima and local
    #   minima around each respective pixel.
    img_min = ndimage.minimum_filter(input_img, footprint=sk.morphology.disk(disksize_range))
    img_max = ndimage.maximum_filter(input_img, footprint=sk.morphology.disk(disksize_range))
    img_delta = img_max - img_min    
        # plt.imshow(img_delta)
    
    # Use an otsu threshold to select regions with high local differences (i.e. bacteria)
    thresh = sk.filters.threshold_otsu(img_delta)
    mask_global = img_delta > thresh
        # plt.imshow(mask_global)
    
    # Fill holes
    # - Above works, but resulting mask has holes. Fill those.
    mask_global_fill = ndimage.binary_fill_holes(mask_global)
    
    # Select largest region (removes artifacts)
    labeled_global = sk.measure.label(mask_global_fill)
    largest_region = np.argmax([r.area for r in sk.measure.regionprops(labeled_global)]) + 1
    mask_global_filled = labeled_global == largest_region
        # plt.imshow(mask_global_filled)
       
    # Now use erosion, as the mask is oversize due to neighborhood size     
    border_margin=5 # keep a certain margin still, though
    disksize_shrink = np.max([0, disksize_range-border_margin])
    mask_global_final = \
        sk.morphology.erosion(
            mask_global_filled, 
            footprint=sk.morphology.disk(disksize_shrink)
            )
        # plt.imshow(mask_global_final)    
        
    # Plot the result (debugging only)
    # plt.imshow(input_img); plt.contour(mask_global_final, levels=[0.5], colors='r')
    
    # now also obtain related binding box
    bbox = sk.measure.regionprops(mask_global_final.astype(int))[0].bbox
    
    return mask_global_final, bbox
    
def _get_edges(input_img_crop, sigma_smooth, disksize_crossing, showplot=False):
    """ Find edges in an image using Laplacian of Gaussian appraoch. """

    # Smooth image before Laplacian filtering
    img_gauss = sk.filters.gaussian(input_img_crop, sigma=sigma_smooth)
    img_laplacian = sk.filters.laplace(img_gauss)
    
    if showplot:
        minmaxval = np.min([-np.min(img_laplacian), np.max(img_laplacian)])
        plt.imshow(img_laplacian, cmap='bwr', vmin=-minmaxval, vmax=minmaxval) 
        plt.show()
        # plt.hist(img_laplacian.ravel(), bins=100)

    # Identify LoG zero crossings as bacterial outlines
    edges_min = ndimage.minimum_filter(img_laplacian, footprint=sk.morphology.disk(disksize_crossing))
    edges_max = ndimage.maximum_filter(img_laplacian, footprint=sk.morphology.disk(disksize_crossing))
    mask_edges = np.logical_and(edges_min < 0, edges_max > 0)
    
    if showplot:
        plt.imshow(mask_edges, cmap='gray')
    
    return mask_edges
    

def _bacterial_seeds(input_img_crop, min_distance=5, showplot=False):
    """ Identify bacterial locations using local maxima. 
    
    This simple function only aims to locations that cover the insides
    of all bacteria, but multiple locations per bacteria are OK.
    """
    
    # apply gaussian
    img_smooth = sk.filters.gaussian(input_img_crop, sigma=3)

    # get local maxima
    seed_locations = sk.feature.peak_local_max(img_smooth, min_distance=min_distance)
    
    # plot the result
    if showplot:
        plt.imshow(input_img_crop)
        for x,y in seed_locations:
            plt.plot(y,x,'ro')
        plt.show()
    
    return seed_locations

def _flood_fill_multiple(mask_edges, seed_locations):
    
    # Create an empty mask to hold the filled regions
    mask_filled = mask_edges.copy()
    
    # Loop over each seed location and perform flood fill
    for x, y in seed_locations:
        # (x, y) = seed_locations[0]

        mask_filled = \
            sk.morphology.flood_fill(
                mask_filled, 
                (x, y),
                2)
    
    # plt.imshow(mask_filled)
    
    return mask_filled
   
def _unique_seeds_erosion(mask_bacteria, disksize_bacshrink=5):   
    """ Aims to obtain seeds that correspond to unique indivual bacteria. """
    
    mask_bacteria_shrunk = \
        sk.morphology.erosion(mask_bacteria, 
                              footprint=sk.morphology.disk(disksize_bacshrink))
                              
    return sk.measure.label(mask_bacteria_shrunk)
    
def _unique_seeds_distance(mask_bacteria, sigma_seed, distance_threshold):
    # sigma_seed=3; distance_threshold=3
    
    # create distance map    
    distance_map = ndimage.distance_transform_edt(mask_bacteria)
    # median filter on distance map
    distance_map_blur = sk.filters.gaussian(distance_map, sigma=sigma_seed)
        # plt.imshow(distance_map_blur, cmap='jet')

    mask_seeds = distance_map_blur > distance_threshold
        # plt.imshow(mask_seeds)
        
    return sk.measure.label(mask_seeds)
        

    
def seg_bacterium(input_img, sigma_smooth = 3, 
                    disksize_crossing=1, disksize_range=20, min_distance=5,
                    sigma_seed=3, distance_threshold=3):
                    # sigma_seed=2, distance_threshold=4):
    """
    Segment bacteria in an image using LoG zero crossings and watershed.
    """
    # input_img = invert(imgs_ecoli[10,:,:])
    # input_img = invert(imgs_ecoli[100,:,:])
    # input_img = invert(imgs_ecoli[1000,:,:])
        # plt.imshow(input_img)
    # sigma_smooth = 3; disksize_crossing=1; disksize_range=20; min_distance=5; sigma_seed=2; distance_threshold=5
    
    mask_global, bbox = _global_mask(input_img, disksize_range=disksize_range)
        # plt.imshow(mask_global)
    
    # crop the image for speed
    input_img_crop   = input_img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    mask_global_crop = mask_global[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        # plt.imshow(mask_global_crop)
    
    mask_edges  = _get_edges(input_img_crop, sigma_smooth = sigma_smooth, disksize_crossing=disksize_crossing)
        # plt.imshow(mask_edges)
    
    seed_locations = _bacterial_seeds(input_img_crop, min_distance=min_distance)
    mask_filled = _flood_fill_multiple(mask_edges, seed_locations)
        # plt.imshow(mask_filled)
    
    mask_bacteria = np.logical_and(mask_filled, mask_global_crop)
        # plt.imshow(mask_bacteria)
    
    mask_unique_seeds = \
        _unique_seeds_distance(mask_bacteria, 
                               sigma_seed=sigma_seed, 
                               distance_threshold=distance_threshold)
        # plt.imshow(mask_unique_seeds)
    
    # now apply watershed
    result = sk.segmentation.watershed(-1 * mask_bacteria, markers=mask_unique_seeds, mask=mask_bacteria)
        # plt.imshow(result, cmap=)    
        # plt.imshow(result, cmap=cmap_random(200))
            
    return result, input_img_crop
    
def plot_sidebyside(img_inv_crop, segmask_oneframe, segmask_oneframe):

    # set font to arial 8
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8

    # create figure
    fig, axs = plt.subplots(1, 2, figsize=(10/2.54, 5/2.54))

    axs[0].imshow(img_inv_crop, cmap='gray')
    axs[0].title(f"Frame {frame_idx}")
    axs[1].imshow(segmask_oneframe, cmap=cmap_random(200))
    axs[1].title(f"Segmentation")
    
    return fig

################################################################################
# %% Apply it


# This statements checks if the script is being run directly
if __name__ == "__main__":

    # read config file
    config_file_path = "answers/bacteria_config_example.yaml"
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # set up output folder
    basename = config['path'].split('/')[-1].split('.')[0]
    output_subfolder = os.path.join(config['path_output'], basename)
    # create output sub folder
    os.create_dirs(output_subfolder, exist_ok=True)
    os.create_dirs(os.path.join(output_subfolder, "plots"), exist_ok=True)    
    # write the configuration file to the subfolder
    yaml.dump(config, open(os.path.join(output_subfolder, 'config_used.yaml'), 'w'))
    
    # Load data
    imgs_ecoli = tiff.imread(config['path'])
    
    # loop over frames
    for frame_idx in range(config['frame_range'][0], config['frame_range'][1]):
    
        # now segment a frame
        segmask_oneframe, img_inv_crop = seg_bacterium(
            invert(imgs_ecoli[frame_idx,:,:]), 
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
                fig = plot_sidebyside(img_inv_crop, segmask_oneframe, segmask_oneframe)
                fig.savefig(
                    os.path.join(
                        output_subfolder, "plots/", 
                        f"png_segmentation_frame{frame_idx:04d}.png"
                        ), 
                    dpi=600
                    )
    
        # save it to npz in output folder
        output_path = f"{config['path_output']}segmentation_frame{frame_idx:04d}.npz"
        np.savez(
            output_subfolder, 
            segmask=segmask_oneframe,
            img_inv_crop=img_inv_crop
            )
# %%
