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

f_nco = 3.9 *1e9  # dummy variable

"""start stop"""
f_start = 4.4* 1e9
f_stop = 4.44* 1e9

"""center/span"""
_f_center=4.408e9
_f_span=10e6
f_start,f_stop=_f_center+_f_span*np.array([-1/2,1/2])

df = 20_000
f_delta = 20_000
nr_pxs = 1000
f_arr = np.arange(f_start, f_stop + f_delta, f_delta)
nr_freq = len(f_arr)
#
# v_amp_arr = np.linspace(0.1,1.0,10)
v_amp_arr = np.linspace(0.03,0.6,20)
# v_amp_arr= np.array([0.02, 0.05, 0.1, 0.5, 0.9])
# v_amp_arr = np.logspace(-2,0,11)
# v_amp_arr= np.ones(25)*0.05
# v_amp_arr = np.array([0.05, 0.5])
#
nr_amps = len(v_amp_arr)

##############################################################################
# Set measurement type

reverse = False
fwd_and_rev = False

###########################################################################


###########################################################################
# Instantiate a Presto Lockin-object in mixed mode

with lockin.Lockin(address=BOX_ADDRESS, **CONVERTER_RATES, port = PORT) as lck:
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

    # Configure the NCO for pump frequencies
    lck.hardware.configure_mixer(
        freq=f_nco,
        in_ports=v_input_port,
        out_ports=v_output_signal_port
        # tune=False
    )
    print("NCO set")

    # Create a lockin output group object, for one frequency, upper sideband only
    outgroup = lck.add_output_group(v_output_signal_port, 1)
    # outgroup_jpa = lck.add_output_group(JPA_port, 1)
    # outgroup_jpa.set_amplitudes(0.)
    # outgroup.set_phases(phases=0.0, phases_q=-np.pi / 2)
    outgroup.set_phases(phases=0.0, phases_q=-np.pi / 2)

    # Create a lockin input group object, with for one frequencies
    ingroup = lck.add_input_group(v_input_port, 1)
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
    # n_start = int(round(f_start / df))
    # n_stop = int(round(f_stop / df))
    # n_arr = np.arange(n_start, n_stop + 1)
    # nr_freq = len(n_arr)
    # freq_arr = df * n_arr
    # freq_arr=np.arange(f_start,f_stop+1, f_delta)
    # nr_freq=len(freq_arr)

    # # Raise error if offset exceeds the Vivace's bandwidth
    # f_offset = freq_arr[-1] - f_nco
    # if f_offset > 500e+6:
    #     raise ValueError("f_offset is too large. It is {} Hz".format(f_offset))

    # # Shift
    # freqs = freq_arr - f_nco

    # # Tuneta
    # f_arr, df = lck.tune(freqs, df)

    # # Adjust for sweep direction(s)
    # if fwd_and_rev:
    #     f_arr = np.concatenate((f_arr, np.flip(f_arr)))
    #     print ("Forward and reverse direction")
    # elif reverse:
    #     f_arr = np.flip(f_arr)
    #     print ("Reverse direction")
    # else:
    #     print("Forward direction")

    #######################################################################
    # Apply df
    lck.set_df(df)
    lck.apply_settings()
    time.sleep(0.1)

    # Verify
    print(f"Applied df: {lck.get_df()} Hz")

    #######################################################################
    # Start measurement

    # Create empty index dictionary
    idx_dct = dict(idx=0, v_amp=0.0)

    # Progressbar
    with tqdm(total=(nr_amps * nr_freq), ncols=80) as pbar:
        # Loop over amplitudes
        for v_idx, v_amp in enumerate(v_amp_arr):
            # Apply pump amplitude
            outgroup.set_amplitudes(v_amp)
            lck.apply_settings()
            time.sleep(0.1)

            # update attributes
            idx_dct["idx"] = v_idx
            idx_dct["v_amp"] = v_amp

            # response array
            resp_arr = np.zeros(nr_freq, dtype=np.complex128)

            # Verify
            # print("EM amplitudes applied: {}".format(outgroup.get_amplitudes()))

            # Iterate over frequency array
            for f_idx, f in enumerate(f_arr):
                # Set frequency
                lck.hardware.configure_mixer(
                    freq=f,
                    in_ports=v_input_port,
                    out_ports=v_output_signal_port
                    # tune=False
                )
                outgroup.set_frequencies(0)
                ingroup.set_frequencies(0)
                lck.apply_settings()
                time.sleep(0.0001)

                discarded_pxs = 1

                # Get lockin packets
                data = lck.get_pixels(nr_pxs + discarded_pxs)

                # Extract data
                freqs, pixels_I, pixels_Q = data[v_input_port]

                # Downconvert (ignore first pixel)
                LSB, HSB = utils.untwist_downconversion(
                    pixels_I[discarded_pxs:, 0], pixels_Q[discarded_pxs:, 0]
                )

                # Append the data to the arrays
                resp_arr[f_idx] = np.mean(HSB)

                # Progressbar update
                pbar.update(1)

            #######################################################################
            # Save data

            # Save the data with attributes
            save_data(save_folder, save_file, sample, myrun, f_arr, resp_arr, idx_dct)

    #######################################################################
    # RF off
    outgroup.set_amplitudes(0.0)  # amplitude [0.0 - 1.0]
    lck.apply_settings()
    time.sleep(0.1)
    lck.hardware.set_dc_bias(0.0, 1)

    # Read applied settings
    print("\n")
    print("# *** Optical pump ******* #")
    print(f"Ampl: {outgroup.get_amplitudes()} FS")
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
    "nr_amps": nr_amps,
    "Presto API": presto.__version__,
    "Npixels": nr_pxs,
    # "Npoints": nr_freq,
    "NCO": f_nco,
    "t_start": t_start,
    "t_end": t_end,
    "Direction": None,
    "Stepping": "v_amp",
    "Script name": os.path.basename(__file__),
}

# Assign sweep direction
if reverse:
    myrun_attrs["Direction"] = "Reverse"
elif fwd_and_rev:
    myrun_attrs["Direction"] = "Forward and Reverse"
else:
    myrun_attrs["Direction"] = "Forward"

# Save script and run attributes
save_script(save_folder, save_file, sample, myrun, myrun_attrs)

# Debug
print("Finished on:", time.strftime("%Y-%m-%d_%H_%M_%S"))
print(f"Run name: {myrun}")
print("Done")
