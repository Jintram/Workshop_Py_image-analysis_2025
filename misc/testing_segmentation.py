
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
    # save to desktop
    plt.savefig('/Users/m.wehrens/Desktop/my_plot_12.png', dpi=300)

# %%

# Let's also load an image of some bacteria (Wehrens et al.)
path_img_ecoli = '../images/biological/microcolony_ecoli.tif'
img_ecoli = tiff.imread(path_img_ecoli)
img_ecoli_inv = sk.util.invert(img_ecoli)

# perform rolling ball background correction (radius )
background = sk.restoration.rolling_ball(img_ecoli_inv, radius=25)
background_smooth = sk.filters.gaussian(background, sigma=3)
img_ecoli_inv_bgcor = img_ecoli_inv - background_smooth

my_plot_12(background, background_smooth, mycmap='gray')
plt.imshow(background_smooth)
my_plot_12(img_ecoli_inv, img_ecoli_inv_bgcor, mycmap='viridis')

# %%

# Show the results of LoG on the bacteria
img_ecoli_gauss = sk.filters.gaussian(img_ecoli_inv, sigma=3) # 3
edges_log = sk.filters.laplace(img_ecoli_gauss, ksize=3)
#edges_logl = sk.filters.laplace(edges_log, ksize=5)

# show both
my_plot_12(img_ecoli_gauss, edges_log, mycmap='viridis')
plt.imshow(edges_log, cmap='seismic')
plt.savefig('/Users/m.wehrens/Desktop/my_plot_3.png', dpi=300)

# show histogram of edges_log
plt.hist(edges_log.ravel(), bins=100, color='gray')
# %%
