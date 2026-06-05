




# %%

import imageio as iio
import matplotlib.pyplot as plt
import numpy as np

# %%


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
        
        # return result & figure
        return img_corr, fig
        
    # if no export path, only return result
    return img_corr, None

# %%

if __name__ == "__main__":
    
    # some example code
    
    path_KTR_example = '/Users/m.wehrens/Data_notbacked/2025_Py-Image-workshop_KTR-example-data/raw/Composite_KTR.tif'
    img_KTR = iio.imread(path_KTR_example)[0, 2, 0:200, 0:200]
    
    img_KTR_corr, _ = correct_bg(img_KTR)
    plt.imshow(img_KTR_corr)

