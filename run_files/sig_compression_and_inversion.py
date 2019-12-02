"""
Script orchestrating running of sig compression of data and its inversion using algo in
"Inverting the signature of a path", https://arxiv.org/abs/1406.7833
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
import numpy as np
np.set_printoptions(threshold=sys.maxsize)  # allows to print more elements in the line
import base.inverse_sig_computer
from base.auto_encoders import auto_encoder_shallow
from data.data_preparation import data_set_1


# 1. Getting data and transforming into sig
handwritting_ds = data_set_1(sorting_data=True)
sig_data_train = np.float32(iisignature.sig(handwritting_ds.training_X, 5))
sig_data_test = np.float32(iisignature.sig(handwritting_ds.test_X, 5))

shape_single_data_point = sig_data_train[0].shape

# 2. Setting and training auto-encoders
auto_encoder = auto_encoder_shallow(encoding_dim=5, input_shape=shape_single_data_point)
auto_encoder.train(training_set=sig_data_train, test_set=sig_data_test, epochs=2)

# 3. Plotting difference between training
encoded_data = auto_encoder.encoder.predict(sig_data_test)
decoded_data = auto_encoder.decoder.predict(encoded_data)

print("sig_data_test")
# print(sig_data_test)
print(type(sig_data_test[0][0]))

print("encoded data")
print(encoded_data)
print(encoded_data.shape)
print(type(encoded_data))

print("decoded_data")
print(decoded_data)
print(decoded_data.shape)
print(type(decoded_data))

# pprint("difference between both")
# pprint(sig_data_test - decoded_data)
