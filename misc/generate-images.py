
# %%

# based on https://bioimagebook.github.io/chapters/2-processing/3-thresholding/thresholding.html

import numpy as np
import tifffile as tiff
from scipy import ndimage
from skimage import exposure

# %% Modify the happy_cell image, add noise

noise_sigma=5
seed=2048

im = tiff.imread("images/bioimagebook/happy_cell.tif")

rng = np.random.default_rng(seed)
im = im + rng.normal(scale=noise_sigma, size=im.shape)
# rescale whole range to 0-255, np.uint8, using 

im = exposure.rescale_intensity(im, out_range='uint8')

tiff.imwrite("images/bioimagebook/happy_cell_noise.tif", im)

# %% Create the spot image

def create_spots(shape=(400, 400), n_spots=10, spot_sigma=4, spot_intensity=5, seed=1024, to_uint8=True):
    im = np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, high=im.shape[0], size=n_spots)
    cols = rng.integers(0, high=im.shape[1], size=n_spots)
    im[rows, cols] = 100
    im = ndimage.gaussian_filter(im, sigma=spot_sigma)
    im = im / im.max() * spot_intensity
    im = im + rng.normal(size=im.shape)
    im = exposure.rescale_intensity(im, out_range='uint8')
    return im
    
im = create_spots() 

# plt.imshow(im, cmap='grey')
tiff.imwrite("images/bioimagebook/spots.tif", im)
    
# (..)    

# %%
