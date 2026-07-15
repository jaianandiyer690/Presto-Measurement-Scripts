# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:23:50 2025

@author: Ermesc & Nils
"""

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


def save_data(folder, file, sample, myrun, freq_arr, sig_data, idl_data, index_dct):
    destination_folder=os.path.join(folder,file,sample)
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(destination_folder, myrun+".hdf5"), "a") as savefile:
        # Determine index number under run
        idx = index_dct["idx"]

        # String as handles
        freq_data_str = f"{str(idx)}/Frequency"
        sig_data_data_str = f"{str(idx)}/Signal_Data"
        idl_data_data_str = f"{str(idx)}/Idler_Data"
        idx_attrs_str = f"{str(idx)}"

        # Save the data arrays
        savefile.create_dataset(freq_data_str, (np.shape(freq_arr)), dtype=float, data=(freq_arr))
        savefile.create_dataset(sig_data_data_str, (np.shape(sig_data)), dtype=complex, data=(sig_data))
        savefile.create_dataset(idl_data_data_str,  (np.shape(idl_data)), dtype=complex, data=(idl_data))

        savefile[freq_data_str].attrs["Unit"] = "Hz"
        savefile[sig_data_data_str].attrs["Unit"] = "fsu complex"
        savefile[idl_data_data_str].attrs["Unit"] = "fsu complex"

        # Write index attributes
        for k, v in index_dct.items():
            savefile[idx_attrs_str].attrs[k] = v


###########################################################################
# Specify save folder location, save file name and run name in "parameter_file.py"

myrun = time.strftime("%Y-%m-%d_%H_%M_%S")

###########################################################################
# Set pump parameters



"""start stop"""
# f_start = 4.1* 1e9
# f_stop = 5.0* 1e9

"""center/span"""
_f_center= 4426539368.473921  #From fitting resonance at v_dc = 2.0
_f_span=2.5e6
f_start,f_stop=_f_center+_f_span*np.array([-1/2,1/2])

f_nco_pump = 2 * _f_center

df = 20_000
f_delta = 20_000
nr_pxs = 1

#
v_amp_signal = 0.01

v_dc = 4.0

v_amp_pump_arr = np.linspace(0.0, 0.0, 6)
# v_amp_pump_arr = np.array([0, 0.1])

###########################################################################
# Instantiate a Presto Lockin-object in mixed mode

with lockin.Lockin(address=BOX_ADDRESS, **CONVERTER_RATES, port=PORT) as lck:
    # Start time
    t_start = time.strftime("%Y-%m-%d_%H_%M_%S")

    # Print "myrun"
    print(f"Run: {myrun}")

    # Print Presto version
    print("Presto version: " + presto.__version__)

    # Print temperature
    print(f"Temperature set: {temperature * 1000.0} mK")

    #######################################################################
    # Setup the input and output group for EM resonator, as well as the
    # frequency of the NCO used to drive the EM resonator.

    
    print("signal NCO set")

     # Configure the NCO for pump frequencies
    lck.hardware.configure_mixer(
        freq=f_nco_pump,
        out_ports=v_output_pump_port
        # tune=False
    )

    # Create a lockin output group object, for one frequency, upper sideband only
    outgroup_signal = lck.add_output_group(v_output_signal_port, 1)
    outgroup_signal.set_phases(phases=0.0, phases_q=-np.pi / 2)
    outgroup_signal.set_amplitudes(v_amp_signal)

     # Create a lockin output group object, for one frequency, upper sideband only
    outgroup_pump = lck.add_output_group(v_output_pump_port, 1)
    outgroup_pump.set_phases(phases=0.0, phases_q=-np.pi / 2)
    
    # Create a lockin input group object, with for two frequencies
    ingroup = lck.add_input_group(v_input_port, 2)
    ingroup.set_phases(0.0)
    lck.apply_settings()
    time.sleep(0.1)

    #######################################################################
    # Prepare data arrays

    # Riccardo's method
    fs = lck.get_fs("dac")
    print(f"Sampling frequency fs: {fs:.3e} Hz")
    nr_samples = int(round(fs / df))
    df = fs / nr_samples
    #######################################################################
    # Apply df
    lck.set_df(df)
    lck.apply_settings()
    time.sleep(0.1)

    # Verify
    print(f"Applied df: {lck.get_df()} Hz")

    #######################################################################
    nr_freq = int(_f_span/df)
    if nr_freq%2 == 0:
        nr_freq += 1
    f_arr = np.arange(nr_freq)*df
    f_nco_signal = _f_center - f_arr[int(nr_freq/2)]  # 

    print(f"f_nco_signal: {f_nco_signal}")

    # Configure the NCO for signal frequencies
    lck.hardware.configure_mixer(
        freq=f_nco_signal,
        in_ports=v_input_port,
        out_ports=v_output_signal_port
        # tune=False
    )

    # Start measurement

    # Create empty index dictionary
    idx_dct = dict(idx=0, v_pump=0.0)

    # Progressbar
    with tqdm(total=(len(v_amp_pump_arr) * nr_freq), ncols=80) as pbar:
        # Loop over pump amplitudes
        for v_pump_idx, v_pump in enumerate(v_amp_pump_arr):
            lck.hardware.set_dc_bias(bias=v_dc,port=vdc_port,range_i=1) 

            outgroup_pump.set_amplitudes(v_pump)
            
            lck.apply_settings()
            time.sleep(0.1)

            # update attributes
            idx_dct["idx"] = v_pump_idx
            idx_dct["v_pump"] = v_pump

            # response array
            resp_arr_sig = np.zeros(nr_freq, dtype=np.complex128)
            resp_arr_idl = np.zeros(nr_freq, dtype=np.complex128)

            # Verify
            # print("EM amplitudes applied: {}".format(outgroup.get_amplitudes()))

            # Iterate over frequency array
            for f_idx, f in enumerate(f_arr):
                # Set frequency
                outgroup_signal.set_frequencies(f_arr[f_idx])             
                ingroup.set_frequencies([f_arr[f_idx], f_arr[-f_idx - 1]])
                lck.apply_settings()
                time.sleep(0.001)

                discarded_pxs = 1

                # Get lockin packets
                data = lck.get_pixels(nr_pxs + discarded_pxs)

                # Extract data
                freqs, pixels_I, pixels_Q = data[v_input_port]

                # Downconvert (ignore first pixel)
                LSB_sig, HSB_sig = utils.untwist_downconversion(
                    pixels_I[discarded_pxs:, 0], pixels_Q[discarded_pxs:, 0]
                )

                 # Downconvert (ignore first pixel)
                LSB_idl, HSB_idl = utils.untwist_downconversion(
                    pixels_I[discarded_pxs:, 1], pixels_Q[discarded_pxs:, 1]
                )

                # Append the data to the arrays
                resp_arr_sig[f_idx] = np.mean(HSB_sig)
                resp_arr_idl[-f_idx - 1] = np.mean(HSB_idl)


                # Progressbar update
                pbar.update(1)

            #######################################################################
            # Save data

            # Save the data with attributes
            save_data(save_folder, save_file, sample, myrun, f_nco_signal + f_arr, resp_arr_sig, resp_arr_idl, idx_dct)

    #######################################################################
    # RF off
    outgroup_signal.set_amplitudes(0.0)  # amplitude [0.0 - 1.0]
    outgroup_pump.set_amplitudes(0.0) 
    lck.apply_settings()
    time.sleep(0.1)
   
    # DC off
    lck.hardware.set_dc_bias(bias=0.0,port=vdc_port,range_i=1) 

    # Read applied settings
    print("\n")
    print("# *** Optical pump ******* #")
    print(f"Signal Ampl: {outgroup_signal.get_amplitudes()} FS")
    print(f"Pump Ampl: {outgroup_pump.get_amplitudes()} FS")
    print("\n")

#######################################################################
# Save the script and run attributed

# Get time stamp
t_end = time.strftime("%Y-%m-%d_%H_%M_%S")

# Create a dictionary containing the run attributes,
myrun_attrs = {
    "Meas": "Frequency sweep",
    "Instr": Instr,
    "cryostat": cryostat,
    "Box": BOX,
    "Sample": sample,
    "Comment": comment,
    "T": temperature,
    "att": atten,
    "4K_gain": 42,
    "rt_gain_output": rt_gain_output,
    "rt_gain_input": rt_gain_input,
    "rt_amplifier": "",
    "df": df,
    "nr_freq": nr_freq,
    "nr_steps": len(v_amp_pump_arr),
    "Presto API": presto.__version__,
    "Npixels": nr_pxs,
    "NCO_signal": f_nco_signal,
    "t_start": t_start,
    "t_end": t_end,
    "Stepping": "v_amp_pump",
    "v_amp_signal": v_amp_signal,
    "v_dc": v_dc,
    "center_frequency": _f_center,
    "Script name": os.path.basename(__file__),
}


# Save script and run attributes
save_script(save_folder, save_file, sample, myrun, myrun_attrs)

# Debug
print("Finished on:", time.strftime("%Y-%m-%d_%H_%M_%S"))
print(f"Run name: {myrun}")
print("Done")
