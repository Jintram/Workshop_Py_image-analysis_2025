

# %% Sanity check to check coordinates


import matplotlib.pyplot as plt
import numpy as np

# generate 100 x 500 image with noise
img_example = np.random.random((100, 500))
plt.imshow(img_example)

# now set extent explicitly
plt.imshow(img_example, extent=[0,5,0,1])
