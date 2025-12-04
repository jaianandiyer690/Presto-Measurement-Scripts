# -*- coding: utf-8 -*-

import h5py
import os
import time
import numpy as np
from tqdm import tqdm
import inspect

import presto
from presto import lockin, utils

from parameter_file import *

def save_script(folder, file, sample, myrun, myrun_attrs):
    # Required package to determine full filename and path

    destination_folder=os.path.join(folder,file,sample)
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)
    # Determine the full filename, including path, of the script
    filename = inspect.getframeinfo(inspect.currentframe()).filename

    # Read each line of the script (this file)
    with open(filename, "r") as codefile:
        # Save the lines of the script as a list
        code_lines = codefile.readlines()

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(destination_folder, myrun+".hdf5"), "a") as savefile:
        # Define the datatype for the dataset to be string
        dt = h5py.special_dtype(vlen=str)

        # Create a dataset named "source code" in the "run" group
        code_set = savefile.create_dataset(
            f"Source code", (len(code_lines),), dtype=dt
        )

        # Write the lines of the script to the dataset
        for i in range(len(code_lines)):
            code_set[i] = code_lines[i]

        # Save the attributes of the run
        for key in myrun_attrs:
            savefile.attrs[key] = myrun_attrs[key]

    # Debug
    print("Saved script and run attributes")


def save_data(folder, file, sample, myrun, freq_arr, data, index_dct):
    destination_folder=os.path.join(folder,file,sample)
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(destination_folder, myrun+".hdf5"), "a") as savefile:
        # Determine index number under run
        idx = index_dct["idx"]

        # String as handles
        freq_data_str = f"{str(idx)}/Frequency"
        data_data_str = f"{str(idx)}/Data"
        idx_attrs_str = f"{str(idx)}"

        # Save the data arrays
        savefile.create_dataset(freq_data_str, (np.shape(freq_arr)), dtype=float, data=(freq_arr))
        savefile.create_dataset(data_data_str, (np.shape(data)), dtype=complex, data=(data))

        savefile[freq_data_str].attrs["Unit"] = "Hz"
        savefile[data_data_str].attrs["Unit"] = "fsu complex"

        # Write index attributes
        for k, v in index_dct.items():
            savefile[idx_attrs_str].attrs[k] = v


###########################################################################
# Specify save folder location, save file name and run name in "parameter_file.py"

myrun = time.strftime("%Y-%m-%d_%H_%M_%S")

###########################################################################
# Set pump parameters

f_0 = 4.300 * 1e9
f_detuning = 3e6 #change this according to measurements'
_f_pump=f_0-f_detuning

f_nco = 4.205 * 1e9 #Nco freq must be smaller than min(f0-detuning, f0-span/2)

f_span = 5.5 * 1e6

f_start = f_0 - f_span / 2
f_stop = f_0 + f_span / 2

df = 50_000
f_delta = 50_000

nr_pxs = 2000

v_prob = 0.02  # probe amplitude
v_pump_arr = np.linspace(0.0, 0.9, 10)  # pump amplitude
# v_pump_arr = np.array([0.0, 0.02, 0.05, 0.1, 0.3])
nr_amps = len(v_pump_arr)

verbose = False

##############################################################################
# Set measurement type

reverse = False
fwd_and_rev = False

input_port = 1
v_output_signal_port = 1

###########################################################################
# Instantiate a Presto Lockin-object in mixed mode

with lockin.Lockin(address=BOX_ADDRESS, **CONVERTER_RATES, port=PORT) as lck:
    t_start = time.strftime("%Y-%m-%d_%H_%M_%S")
    if verbose:
        print(f"Run: {myrun}")
        print(f"Presto version: {presto.__version__}")
        print(f"Temperature set: {temperature*1e3} mK")
        print(f"Sample: {sample}")
        print(f"Save file: {save_file}")
        # print(f"Pump type: {pump_type}")
        print(f"nr_amps: {nr_amps}")
        print(f"nr_pxs: {nr_pxs}")

    lck.hardware.configure_mixer(
        freq=f_nco,
        in_ports=input_port,
        out_ports=v_output_signal_port,
    )

    og = lck.add_output_group(v_output_signal_port, 2)
    og.set_phases(phases=[0.0, 0.0], phases_q=[-np.pi / 2, -np.pi / 2])

    ig = lck.add_input_group(input_port, 1)
    ig.set_phases(0.0)

    lck.apply_settings()
    time.sleep(0.1)

    #######################################################################
    # Prepare data arrays

    fs = lck.get_fs("dac")
    if verbose:
        print(f"Sampling frequency fs: {fs:.3e} Hz")
    nr_samples = int(round(fs / df))
    df = fs / nr_samples
    
    prob_freq_arr = np.arange(f_start, f_stop + 1, f_delta)
    nr_freq = len(prob_freq_arr)

    # Raise error if offset exceeds the Vivace's bandwidth
    f_offset = np.max([prob_freq_arr[-1],_f_pump]) - f_nco
    if f_offset > 500e6:
        raise ValueError(f"f_offset is too large. It is {f_offset} Hz")

    # Shift
    _freqs = prob_freq_arr - f_nco

    # Tune
    freq_arr, df = lck.tune(_freqs, df)

    # tune mechanical frequency
    f_pump, df = lck.tune(_f_pump-f_nco, df)
    
    og.set_frequencies([f_pump,freq_arr[0]])

    #######################################################################
    # Apply df
    lck.set_df(df)
    lck.apply_settings()
    time.sleep(0.1)

    if verbose:
        print(f"nr_freq: {nr_freq}")
        # print(f"Tuned f_m: {f_m} Hz")
        print(f"Tuned df: {lck.get_df()} Hz")

    #######################################################################
    # Start measurement

    idx_dct = dict(idx=0, pump_amp=0.0)

    with tqdm(total=(nr_amps * nr_freq), ncols=80) as pbar:
        for vv, v_pump in enumerate(v_pump_arr):

            og.set_amplitudes([v_pump, v_prob])
            lck.apply_settings()
            time.sleep(0.1)

            idx_dct["idx"] = vv
            idx_dct["pump_amp"] = v_pump

            resp_arr = np.zeros(nr_freq, dtype=np.complex128)

            for prob_idx, prob_freq in enumerate(freq_arr):
                og.set_frequencies([f_pump,prob_freq])
                ig.set_frequencies(prob_freq)

                lck.apply_settings()
                time.sleep(0.01)

                data = lck.get_pixels(nr_pxs)
                freqs, pixels_I, pixels_Q = data[input_port]
                LSB, HSB = utils.untwist_downconversion(pixels_I[:, 0], pixels_Q[:, 0])

                resp_arr[prob_idx] = np.mean(HSB)

                pbar.update(1)

            save_data(
                save_folder,
                save_file,
                sample,
                myrun,
                f_nco + freq_arr,
                resp_arr,
                idx_dct,
            )

    og.set_amplitudes(0.0)
    lck.apply_settings()
    time.sleep(0.1)


t_end = time.strftime("%Y-%m-%d_%H_%M_%S")

myrun_attrs = {
    "Meas": "Frequency sweep, detuned pump",
    "Instr": Instr,
    "cryostat": cryostat,
    "Box": BOX,
    "Sample": sample,
    "Comment": comment,
    "T": temperature,
    "att": atten,
    # "pump_type": pump_type,
    "4K_gain": 42,
    "rt_gain_output": rt_gain_output,
    # "output_amplifier": "None",
    "rt_gain_input": rt_gain_input,
    # "input_amplifier": "LNF_LNC4_8C",
    "df": df,
    "nr_freq": nr_freq,
    "nr_amps": nr_amps,
    "prob_amp": v_prob,
    "Presto API": presto.__version__,
    "Npixels": nr_pxs,
    "Npoints": nr_samples,
    "NCO": f_nco,
    "f_0": f_0,
    "f_detuning": f_detuning,
    "f_pump" : f_pump,
    "t_start": t_start,
    "t_end": t_end,
    "Direction": "forward",
    "Stepping": "pump_amp",
    "Script name": os.path.basename(__file__),
}


save_script(save_folder, save_file, sample, myrun, myrun_attrs)

if verbose:
    print("Finished on:", time.strftime("%Y-%m-%d_%H_%M_%S"))
    print(f"Run name: {myrun}")
    print("Done")
