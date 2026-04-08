import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
import os
import h5py
import inspect
from tqdm import tqdm
import sys
import math

# Reload credentials module to get latest changes
import importlib

# Timeout for hardware communication
import requests

# BASE URL for interacting with BlueFTC
base_url = 'http://192.168.88.25:5001'

# Print JSON 
def print_JSON(r):
    r_dict = r.json()
    for key, value in r_dict.items():
        print(f"{key}: {value}")


def get_DR_logs(
        timestamp_init,
        timestamp_final,
        channel_nr
):
    # URL endpoint for getting historical data of a channel
    endpoint = 'channel/historical-data' 
    url = base_url + endpoint

    # Channel data request payload
    payload= {
    'channel_nr': channel_nr,
    'start_time': timestamp_init,
    'stop_time': timestamp_final
}
    print("Requesting...", url, '\n')
    r = requests.get(url)

    print('CHANNEL TEMPERATURE \n================')
    print_JSON(r)