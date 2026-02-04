'''
JPA Planck Spectroscopy

This notebook contains the code for running the Planck Spectroscopy experiment on the JPA 
to obtain its noise-temperature graph using an automated temperature ramp-up.
'''


# Import necessary libraries
# ==========================
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import h5py
import inspect
from tqdm import tqdm
import sys
import math
import presto
from presto import lockin, utils, hardware
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode

# Connect to Presto Hardware
# ==========================
ADDRESS = '130.237.35.90'   # IP Address
PORT    = 42873             # TCP Port
Box     = 'Presto DELTA'    # Model Identifier

# Port Assignments on Presto
input_port   = 1 # Output from JPA
output_port  = 1 # Signal to JPA (vacuum for correlation experiments)
flux_port    = 5 # Pump frequency comb 
bias_port    = 4 # DC Bias for optimal operating point of JPA (based on McD curve)

def acquire_planck_data(temp):
    return 0