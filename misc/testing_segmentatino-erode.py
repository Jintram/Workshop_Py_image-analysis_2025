

import matplotlib.pyplot as plt
import seaborn as sns
import tifffile as tiff
import numpy as np

# some new ones
import skimage as sk
from scipy import stats
from scipy import ndimage

image_path = "/Users/m.wehrens/Documents/git_repos/_UVA/_Teaching/2025_teaching_Py_image-analysis/Workshop_Py_image-analysis_2025/images/biological/microcolony_ecoli_intermediate.tif"

# load image file
img_ecoli = tiff.imread(image_path)

# apply an erosion with disk 5
selem = sk.morphology.disk(4)
img_ecoli_eroded = sk.morphology.erosion(img_ecoli, selem)
# plot original and eroded
fig, axs = plt.subplots(1,2, figsize=(10/2.54,5/2.54))
_ = axs[0].imshow(img_ecoli, cmap='viridis')
_ = axs[1].imshow(img_ecoli_eroded, cmap='viridis')
plt.tight_layout()
plt.show()
plt.savefig('/Users/m.wehrens/Desktop/my_plot_eroded.png', dpi=300)
                                         
                                         
# do the same but repeat erosion with disk of 2 twice
selem2 = sk.morphology.disk(2)
img_ecoli_eroded2 = sk.morphology.erosion(img_ecoli, selem2)
img_ecoli_eroded2 = sk.morphology.erosion(img_ecoli_eroded2, selem2)
# plot original and eroded
fig, axs = plt.subplots(1,2, figsize=(10/2.54,5/2.54))
_ = axs[0].imshow(img_ecoli, cmap='viridis')
_ = axs[1].imshow(img_ecoli_eroded2, cmap='viridis')
plt.tight_layout()
plt.show()                                         

# perform morphological opening
img_ecoli_opened = sk.morphology.closing(img_ecoli_eroded2, selem)
# plot original and opened
fig, axs = plt.subplots(1,2, figsize=(10/2.54,5/2.54))
_ = axs[0].imshow(img_ecoli_eroded2, cmap='viridis')
_ = axs[1].imshow(img_ecoli_opened, cmap='viridis')
plt.tight_layout()
plt.show()