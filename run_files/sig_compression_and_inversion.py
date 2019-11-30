"""
Script orchestrating running of sig compression of data and its inversion using algo in
"Inverting the signature of a path", https://arxiv.org/abs/1406.7833
"""

# Build-ins import
import os, sys, inspect

# Home-brew import
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir).replace("run_files", "") + "/src/"
sys.path.insert(0, parent_dir)

import base.sig_computer
import base.inverse_sig_computer
import base.auto_encoders
from data.data_preparation import data_set_1


# 1. Transforming data into signature

