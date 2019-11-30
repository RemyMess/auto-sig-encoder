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
import seaborn as sns

# (importing our modules)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
package_dir = os.path.dirname(current_dir).replace("run_files", "") + "/src"
sys.path.insert(0, package_dir)
import base.sig_computer
import base.inverse_sig_computer
import base.auto_encoders
from data.data_preparation import data_set_1


# 1. Transforming data into signature
handwritting_ds = data_set_1()
sig = iisignature.sig(handwritting_ds.training_X_ordered[1], 3)

# 2. Plotting the sig of all the paths of same category together
sns.set(style="darkgrid")
df = pd.DataFrame(sig[46])

print(df[0])
print(df.index)


plt.show()
#
#
#
# todo
#
