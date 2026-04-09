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
from datetime import datetime

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
    endpoint = '/channel/historical-data' 
    url = base_url + endpoint

    # Channel data request payload
    payload= {
    'channel_nr': channel_nr,
    'start_time': timestamp_init,
    'stop_time': timestamp_final,
    'fields': ['timestamp', 'temperature']  # Specify the fields you want to retrieve
}
    print("Requesting...", url, '\n')
    r = requests.post(url, json=payload)

    timestamp_list = []
    temperature_list = []
    if r.status_code == 200:
        data = r.json()
        # Timestamps
        timestamp_list = data['measurements']['timestamp']        
        datetime_list = [datetime.fromtimestamp(ts) for ts in timestamp_list]

        temperature_list = np.array(data['measurements']['temperature']) * 1e3
    return datetime_list, temperature_list

datetime_list, temperature_list = get_DR_logs(
    timestamp_init='2026-04-02 14:00',  # Example start timestamp
    timestamp_final='2026-04-07 03:00',  # Example end timestamp (1 day later)
    channel_nr=1  # Example channel number
)

# Saving the data to a text file
import pandas as pd

df = pd.DataFrame({
    "timestamp": datetime_list,
    "temperature": temperature_list
})

df.to_csv("Presto-Measurement-Scripts/Planck/DR_logs_channel_1.csv", index=False)

# Plotting the data

plt.figure(figsize=(10, 5))
plt.plot(datetime_list, temperature_list, ls='-', lw=0.5, color='blue')
plt.title('Temperature vs Time')
plt.xlabel('Timestamp')
plt.ylabel('Temperature (mK)')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()



