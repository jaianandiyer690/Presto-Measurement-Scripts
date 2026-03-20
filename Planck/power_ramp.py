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
import presto
from presto import lockin, utils, hardware
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from blueftc.BlueforsController import BlueFTController

# Reload credentials module to get latest changes
import importlib
import JPA_credentials
importlib.reload(JPA_credentials)
from JPA_credentials import API_KEY, IP_ADDRESS, PORT_NUMBER, MXC_ID, HEATER_ID, PID_CALIB_FILE

# Import scripts for JPA AND LKIPA planck spectroscopy
#import JPA_planck_spec 
#import LKIPA_planck_spec

# Timeout for hardware communication
import requests

# BASE URL for interacting with BlueFTC
base_url = 'http://192.168.88.25:5001'

# Print JSON 
def print_JSON(r):
    r_dict = r.json()
    for key, value in r_dict.items():
        print(f"{key}: {value}")

# Get stabilized temperature
# ==========================

def get_stable_temp(N_runs):


    stability_status = False

    runs = np.arange(
        start=1,
        stop=N_runs + 1,
        step=1
    )

    temp_trace = np.full(len(runs), np.nan)

    # Start running temperature plot
    # ==============================

    fig, ax = plt.subplots(
        figsize=(6, 6),
        dpi=125,
    )

    line, = ax.plot(
        [],
        [],
        label='$T_{MXC}$',
        marker='o'
    )

    # horizontal stability lines
    mean_line, = ax.plot([], [], '--', lw=1.0, label='Mean of last 10')
    upper_line, = ax.plot([], [], ':', lw=1.0, label=r'Mean + $\sigma$')
    lower_line, = ax.plot([], [], ':', lw=1.0, label=r'Mean - $\sigma$')

    ax.set_xlabel('Run')
    ax.set_ylabel('Temperature (mK)')
    ax.set_xlim(0, N_runs)
    ax.grid()
    ax.legend()

    display(fig)

    for i in range(len(runs)):

        # Reset channel index and wait until channel 6 measurement is received
        channel = 1
        j = 0
        while channel != 6:

            # Get channel measurement
            url = base_url + '/channel/measurement/latest'
            if j == 0:
                print("Requesting T_MXC")
            r = requests.get(url)
            r_dict = r.json()

            # Retry if wrong channel
            channel = r_dict['channel_nr']
            if channel != 6:
                print('Waiting for Channel 6...')
                time.sleep(2)

            j += 1

        # Sleep until next measurement cycle
        time.sleep(10)

        # Get T_MXC
        temp_trace[i] = r_dict['temperature'] * 1e3  # mK

        # Stability criterion
        n_stable = 10

        if i + 1 < n_stable:
            L5_temp_mean = np.nan
            stability = np.nan

            # keep stability lines hidden until 10 points exist
            mean_line.set_data([], [])
            upper_line.set_data([], [])
            lower_line.set_data([], [])

        else:
            # last 10 temperature readings
            L5_temp_trace = temp_trace[i - n_stable + 1: i + 1]
            stability = np.std(L5_temp_trace)
            L5_temp_mean = np.mean(L5_temp_trace)
            stability_threshold = 0.005 * L5_temp_mean

            # draw horizontal lines across the whole plot
            x_span = [runs[0], runs[-1]]
            mean_line.set_data(x_span, [L5_temp_mean, L5_temp_mean])
            upper_line.set_data(x_span, [L5_temp_mean + stability, L5_temp_mean + stability])
            lower_line.set_data(x_span, [L5_temp_mean - stability, L5_temp_mean - stability])

        # Update temperature trace
        line.set_data(runs[:i+1], temp_trace[:i+1])

        ax.relim()
        ax.autoscale_view(scaley=True)

        clear_output(wait=True)

        print(f'T_MXC = {temp_trace[i]:.3f} mK')
        if i + 1 >= n_stable:
            print(f'Mean(last {n_stable}) = {L5_temp_mean:.3f} mK')
            print(f'STD(last {n_stable}) = {stability:.4f} mK')

            if stability > stability_threshold:
                print('Temperature not stabilized')
            else:
                print('Temperature stabilized')
        else:
            print(f'Waiting for {n_stable} points...')

        display(fig)

        # return once stabilized
        if i + 1 >= n_stable and stability <= stability_threshold:
            stability_status = True
            plt.close(fig)
            return np.round(L5_temp_mean, 2), stability_status

    plt.close(fig)
    return np.round(L5_temp_mean, 2) if not np.isnan(L5_temp_mean) else None, stability_status

# Set heater power, get stable temperature, run PS scripts

def get_stable_temp_ON(heater_power, N_runs):

    # Set heater power to X µW
    # ========================
    url = base_url + '/heater/update'

    heater_nr = 4 # 1, 2, 3, 4 (Mixing Chamber)
    payload = {
        "heater_nr": heater_nr,
        "active": True,
        "pid_mode": 0,
        "power": heater_power
    }

    print("Requesting...", url, '\n')
    r = requests.post(url, json=payload)
    print('Heater power successfully set to ', payload['power'] * 1e6, 'µW.')

    # Wait until temperature stabilized
    stable_temp = get_stable_temp(N_runs=N_runs)

    # Print stabilized temperature
    print('MXC heater power:',str(np.round(heater_power*1e6, 1)), 'microWatt')
    print('T_MXC:', stable_temp, 'milliKelvin')

    # Show heater status
    url = base_url + '/heater'

    heater_nr = 4 # 1, 2, 3, 4 (Mixing Chamber)
    payload= {
        'heater_nr': heater_nr
    }

    r = requests.post(url, json=payload)

    print(f'HEATER {heater_nr} INFORMATION')
    print('=======================')
    print_JSON(r)

    return stable_temp

def heater_ramp_up(MXC_heater_power_list, N_runs):
    for heater_power in MXC_heater_power_list:
        current_temp = get_stable_temp_ON(heater_power=heater_power, N_runs=N_runs)
        print('Running JPA Planck Spectroscopy for T = ', str(np.round(current_temp, 2)))
        # run jpa script
        print('Running LKIPA Planck Spectroscopy for T = ', str(np.round(current_temp, 2)))
        # run lkipa script
        print('Running LKIPA Resonance for T = ', str(np.round(current_temp, 2)))
        # run lkipa resonance script

        # Data saved for current temperature
        print('Data saved for T = ', str(np.round(current_temp, 2)))