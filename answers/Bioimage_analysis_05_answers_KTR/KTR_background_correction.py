

# %% WORK IN PROGRESS


# IMPLEMENT THIS AS FOLDABLE ANSWER TO THE QUESTION

import matplotlib.pyplot as plt
import numpy as np


# OPEN A KTR EXAMPLE IMAGE HERE!!

plt.imshow(img_KTR)


# get the mode
mode_KTR = np.bincount(img_KTR.ravel()).argmax()
mask_bg  = img_KTR<mode_KTR

# display what would count as background
fig, axs = plt.subplots(1,2)
axs[0].imshow(img_KTR)
axs[1].imshow(img_KTR)
axs[1].imshow(mask_bg, alpha=mask_bg*1.0)

# show it in the histogram
plt.hist(img_KTR.ravel(), bins=32, color='grey')
plt.axvline(mode_KTR, color='red')
plt.xlabel("Intensity"); plt.ylabel("Counts")



# NOW WRITE A FUNCTION THAT SUBTRACTS THE BACKGROUND
# (AND SHOW IT)