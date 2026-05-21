

################################################################################
# %% Import libraries

import sys, os
from pathlib import Path
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

################################################################################
# %% plotting function

def plot_data(DATA_PATH, stim_time = None):
    
    # Get relevant path information
    OUTPUT_DIR = Path(DATA_PATH).parent
    FILENAME   = Path(DATA_PATH).stem

    # Read the data
    df_KTR = pd.read_csv(DATA_PATH)

    # add time field
    # From the image metadata: Frame interval: 270.01617 sec
    df_KTR['time_s'] = 270.01617 * df_KTR['frame'] / 60

    # Now plot the data
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 8
    fig, axs = plt.subplots(1,1, figsize=(5/2.54,5/2.54))
    sns.scatterplot(df_KTR, x="time_s", y="KTR_ratio", ax=axs, color="k", alpha=0.5, s=10)
    sns.lineplot(df_KTR, x="time_s", y="KTR_ratio", estimator="mean", color="r", ax=axs)
    axs.axvline(stim_time, color="b", ls="--")
    axs.set_xlabel("Time (min)")
    axs.set_ylabel("Sensor C/N ratio")
    axs.set_ylim(0, 1.5)
    plt.tight_layout()

    # save to same directory
    output_path_fig = os.path.join(OUTPUT_DIR, f"{FILENAME}_overtime.pdf")
    fig.savefig(output_path_fig, dpi=600)

    plt.close(fig)


################################################################################        
# %%


DATA_PATH_ch1 = "/Users/m.wehrens/Data_notbacked/2025_Py-Image-workshop_KTR-example-data/analysis/KTR_ratios_ch1.csv"
plot_data(DATA_PATH_ch1, stim_time=25)

DATA_PATH_ch2 = "/Users/m.wehrens/Data_notbacked/2025_Py-Image-workshop_KTR-example-data/analysis/KTR_ratios_ch2.csv"
plot_data(DATA_PATH_ch2, stim_time=25)

# %%
