import numpy as np
import time
import os
import h5py
import inspect
from tqdm import tqdm

import presto
from presto import lockin, utils
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode


############################################################################
# Saving methods

# Save script function
def save_script(folder, file, sample, myrun, myrun_attrs):
    # Create folders if they do not exist
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # String handles
    run_str = "{}/{}".format(sample, myrun)
    source_str = "{}/{}/Source code".format(sample, myrun)

    # Read lines of the script
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    with open(filename, "r") as codefile:
        code_lines = codefile.readlines()

    # Write lines of script and run attributes
    with h5py.File(os.path.join(folder, file), "a") as savefile:

        dt = h5py.special_dtype(vlen=str)
        code_set = savefile.create_dataset(source_str.format(myrun), (len(code_lines),), dtype=dt)
        for i in range(len(code_lines)):
            code_set[i] = code_lines[i]

        # Save the attributes of the run
        for key in myrun_attrs:
            savefile[run_str].attrs[key] = myrun_attrs[key]

    # Debug
    print("Saved script and run attributes.")


# Save data function
def save_data(folder, file, sample, myrun, freq, usb_arr, lsb_arr, temp):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(folder, file), "a") as savefile:
        # String as handles
        freq_data_str = "{}/{}/freq comb".format(sample, myrun)
        usb_data_str = "{}/{}/USB".format(sample, myrun)
        lsb_data_str = "{}/{}/LSB".format(sample, myrun)
        temp_str = "{}/{}/temperature".format(sample, myrun)

        # Write data to datasets
        savefile.create_dataset(freq_data_str, (np.shape(freq)),
                                dtype=float, data=freq)
        savefile.create_dataset(usb_data_str, (np.shape(usb_arr)),
                                dtype=complex, data=usb_arr)
        savefile.create_dataset(lsb_data_str, (np.shape(lsb_arr)),
                                dtype=complex, data=lsb_arr)
        savefile.create_dataset(temp_str, (np.shape(temp)),
                                dtype=complex, data=temp)

        # Write dataset attributes
        savefile[freq_data_str].attrs["Unit"] = "Hz"
        savefile[usb_data_str].attrs["Unit"] = "fsu complex"
        savefile[lsb_data_str].attrs["Unit"] = "fsu complex"
        savefile[temp_str].attrs["Unit"] = "mK"


# Sample name and total attenuation along measurement chain
sample = 'LKIPA'
meas_type = 'Planck Spectroscopy'


# Lab Network
# ADDRESS = '130.237.35.90'   # from Office
# PORT = 42873                # Presto DELTA
ADDRESS = '192.168.88.53'       # IP Address
PORT    = None                 # TCP Port

# Physical Ports
input_port = 5

# MEASUREMENT PARAMETERS
# NCO frequency
fNCO = 4.42e9
# Bandwidth in Hz
_df = 1e3 # 1e3 for experiment
# Number of pixels
Npix =  2_500_000 # for experiment
N_chunk = 2500  # number of pixels per chunk
# Number of pixels we discard
Nskip = 0
# Number of averages on board
Navg = 1

# SIGNAL PARAMETERS
# Number of frequencies of the frequency comb
nr_sig_freqs = 96 # 96 for experiment
# Frequency span
fs_span = 200e6
# Listening comb
_fs_comb = np.linspace(0, fs_span, nr_sig_freqs)

def get_lkipa_planck(
        save_folder,
        save_file, 
        current_temp
):
    # Saving folder location, saving file and run name
    myrun = time.strftime("%Y-%m-%d_%H_%M_%S")

    # Instantiate lockin device
    with lockin.Lockin(address=ADDRESS,
                    port=PORT,
                    adc_mode=AdcMode.Mixed,
                    adc_fsample=AdcFSample.G2,
                    dac_mode=DacMode.Mixed02,
                    dac_fsample=DacFSample.G6,
                    ) as lck:

        # Tune the listening comb
        fs_comb, df = lck.tune(_fs_comb, _df)

        # Set df
        lck.set_df(df)

        # Data
        usb_arr = np.zeros((Npix, nr_sig_freqs), dtype=np.complex128)
        lsb_arr = np.zeros_like(usb_arr)

        # Configure mixer just to be able to create output and input groups
        lck.hardware.configure_mixer(freq=fNCO,
                                    in_ports=input_port,
                                    )

        # Create input group
        ig = lck.add_input_group(port=input_port, nr_freq=nr_sig_freqs)
        ig.set_frequencies(fs_comb)

        lck.apply_settings()
        lck.hardware.sleep(1e-4, False)

        with tqdm(total=(Npix // N_chunk), ncols=80) as pbar:

            for n in range(0, Npix, N_chunk):

                # Get lock-in packets (pixels) from the local buffer
                data = lck.get_pixels(Nskip + N_chunk, summed=False, nsum=Navg, quiet=True)
                freqs, pixels_i, pixels_q = data[input_port]

                # Convert a measured IQ pair into a low/high sideband pair
                LSB, HSB = utils.untwist_downconversion(pixels_i, pixels_q)

                usb_arr[n:n + N_chunk] = HSB[-N_chunk:]
                lsb_arr[n:n + N_chunk] = LSB[-N_chunk:]

                # Update progress bar
                pbar.update(1)

    # Save data
    save_data(save_folder, save_file, meas_type, myrun, fs_comb + fNCO, usb_arr, lsb_arr, temp=current_temp)


#### TESTING ####
# save_folder = 'D:/Planck Spectroscopy 2026-03/LKIPA'
# save_file = '2026-03-LKIPA-planck_10mk-test.hdf5'
# current_temp = 10

# get_lkipa_planck(
#     save_folder,
#     save_file,
#     current_temp
# )