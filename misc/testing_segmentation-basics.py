


# %%

import matplotlib.pyplot as plt
import seaborn as sns
import tifffile as tiff
import numpy as np

# some new ones
import skimage as sk
from scipy import stats
from scipy import ndimage

# %%

FIGSIZE21 = (10/2.54,5/2.54) 
FIGSIZE22 = (10/2.54,10/2.54) 

def my_plot_12(img1, img2, mycmap='viridis'):
    # plot two images side by side
    fig, axs = plt.subplots(1,2, figsize=FIGSIZE21)
    _ = axs[0].imshow(img1, cmap=mycmap)
    _ = axs[1].imshow(img2, cmap=mycmap)  
    plt.tight_layout()  

# %%

# Let's also load an image of some bacteria (Wehrens et al.)
path_img_photogr = '../images/misc/photographer.tif'
img_photogr = tiff.imread(path_img_photogr)
#img_photogr_inv = sk.util.invert(img_photogr)

# Show the results of LoG on the bacteria
img_photogr_gauss = sk.filters.gaussian(img_photogr, sigma=3) # 3
edges_log = sk.filters.laplace(img_photogr_gauss, ksize=5)

# show both
my_plot_12(img_photogr_gauss, edges_log, mycmap='seismic')
# %%
