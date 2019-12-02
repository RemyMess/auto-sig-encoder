"""
Script generating plots for handwritten number data set: signature and log sig terms
"""


# Build-ins import
import os, sys, inspect
import pandas as pd
import matplotlib.pyplot as plt

# Home-brew import
import iisignature
from pprint import pprint
from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set(style="darkgrid")

# (importing our modules)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
package_dir = os.path.dirname(current_dir).replace("run_files", "") + "/src"
sys.path.insert(0, package_dir)
import base.sig_computer
import base.inverse_sig_computer
# import base.auto_encoders
from data.data_preparation import data_set_1


# 1. Transforming data into signature
handwritting_ds = data_set_1()
truncation_order = 2


def save_plot_sig_logsig_handwritten_data():
    for truncation_order in range(1, 16):
        fig_sig, axs_sig = plt.subplots(10)
        fig_sig.suptitle(f"Sig of handwritten digits (0 top and 9 bottom); k={truncation_order}")

        fig_logsig, axs_logsig = plt.subplots(10)
        fig_logsig.suptitle(f"Log_sig of handwritten digits (0 top and 9 bottom); k={truncation_order}")

        for i in range(10):
            # 1. logsig
            s = iisignature.prepare(2, truncation_order)
            log_sig = iisignature.logsig(handwritting_ds.training_X_ordered[i], s)

            log_sig_df = pd.DataFrame(log_sig)  # signatures are rows
            log_sig_df = log_sig_df.transpose()  # ; we move them into columns so that time series format
            axs_logsig[i].plot(log_sig_df.values)
            fig_logsig.savefig(f"/home/raymess-lin/git/auto-sig-encoder/plots/handwritten_log_sig_k_{truncation_order}.png")

            # 2. sig
            sig = iisignature.sig(handwritting_ds.training_X_ordered[i], truncation_order)

            sig_df = pd.DataFrame(sig)  # signatures are rows
            sig_df = sig_df.transpose()  # ; we move them into columns so that time series format
            axs_sig[i].plot(sig_df.values)
            fig_sig.savefig(f"/home/raymess-lin/git/auto-sig-encoder/plots/handwritten_sig_k_{truncation_order}.png")

# sig
pretrans_data = [[i, 2*i] for i in range(10)]
trans_data = iisignature.sig(pretrans_data, 3)
print(trans_data)

# logsig



plt.plot(trans_data)
plt.savefig("/home/raymess-lin/git/auto-sig-encoder/plots/simple_plot_straight_line_sig_3.png")
