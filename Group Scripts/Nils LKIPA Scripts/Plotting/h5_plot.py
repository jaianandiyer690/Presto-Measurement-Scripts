import h5py
import numpy as np
from scipy.optimize import curve_fit, leastsq, fsolve
from scipy.constants import h, hbar
import matplotlib.pyplot as plt
from matplotlib import interactive
import os
from datetime import datetime

def sort_hdf5_filenames(filenames):
    """
    Sort a list of filenames in the format 'YYYY-MM-DD_HH_MM_SS.hdf5'
    by their embedded date and time.
    """
    def extract_datetime(filename):
        # Remove extension and parse the datetime
        timestamp_str = filename.replace('.hdf5', '')
        return datetime.strptime(timestamp_str, '%Y-%m-%d_%H_%M_%S')
    
    return sorted(filenames, key=extract_datetime)

def correct_delay(S_11_data, meas_freq, delay):
    return S_11_data * np.exp(2*np.pi* 1j * ( meas_freq * delay))

def correct_shift(S_11_data, phi = 0):
    return S_11_data * np.exp(1j * phi)

data_file_path = '/home/kth-user/Documents/data/'
fab_name = 'LKIPA_4'

sample = 'LKIPA_4_08-01'
path = os.path.join(data_file_path,fab_name, sample)

meas_list = os.listdir(path)
meas_list = sort_hdf5_filenames(meas_list)
meas = meas_list[12]

f = h5py.File(os.path.join(path, meas))

measurements = list(f.keys())

print(measurements)

temp = f.attrs["T"]

stepping = f.attrs["Stepping"]
try:
    comment = f.attrs["Comment"]
except: 
    comment = "no comment recorded"
if stepping == "pump_amp":
    delay = 331.5e-9
else:
    delay = 75.5e-9
    delay = 1.07e-7
phi = 0

print(f"Sample name: {sample}")
print(f"Measurment time: {meas}")
print(f"Temperature: {temp}")
print(f"Stepping param: {stepping}")
print(f"Run comment: {comment}")


from matplotlib import colormaps as cm

start_num = 1
stop_num = np.inf
step = 1
num_curves = min(stop_num, len(measurements)-1)
cmap = cm['viridis']
legend = []

save_fig = False
png_title = "v_amp"

ReIm = False
if ReIm:
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
else:
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# fig.tight_layout()

for i in range(start_num, num_curves, step):
    meas_num = i

    S_11_data = f[str(meas_num)]['Data'][()]
    freq = f[str(meas_num)]['Frequency'][()]

    S_11_data_corr = correct_delay(S_11_data, freq, delay)

    if i == start_num:
        phi = -np.angle(S_11_data_corr[0])

    S_11_data_corr = correct_shift(S_11_data_corr, phi)

    if stepping == "v_amp":
        v_amp = f[str(meas_num)].attrs["v_amp"]
        S_11_data_corr_rel = S_11_data_corr/v_amp
        legend.append(f"(i={i})={v_amp:.3f}")
    elif stepping == "vdc":
        S_11_data_corr_rel = S_11_data_corr
        v_dc = f[str(meas_num)].attrs["vdc"]
        legend.append(f"(i={i})={v_dc:.3f}")
    elif stepping == "pump_amp":
        S_11_data_corr_rel = S_11_data_corr
        pump_amp = f[str(meas_num)].attrs["pump_amp"]
        legend.append(f"(i={i})={pump_amp:.3f}")
    else:
        # v_amp = f[str(meas_num)].attrs["v_amp"]
        S_11_data_corr_rel = S_11_data_corr
        # legend.append(f"(i={i})={v_amp:.3f}")


    if num_curves == 1:
        color = cmap(1/2)
    else:
        color = cmap((i - start_num) / (num_curves - start_num - 1))

    if ReIm:
        ax[0,0].plot(freq, 10*np.log10(np.abs(S_11_data_corr_rel)), color = color)
        ax[0,1].plot(freq, np.angle(S_11_data_corr_rel), color = color)

        ax[1,0].plot(freq, np.real(S_11_data_corr_rel), color = color)
        ax[1,1].plot(freq, np.imag(S_11_data_corr_rel), color = color)

        ax[0,0].set_title("Reflection Magnitude |S11|")
        ax[0,0].set_xlabel("Angular Frequency (Hz)")
        ax[0,0].set_ylabel("|S11|")

        ax[0,1].set_title("Reflection Phase ∠S11")
        ax[0,1].set_xlabel("Angular Frequency (Hz)")
        ax[0,1].set_ylabel("Phase (radians)")
        ax[0,1].grid(True, which='both')

        ax[1,0].set_title("Real part of S11")
        ax[1,0].set_xlabel("Angular Frequency (Hz)")
        ax[1,0].set_ylabel("Re{S11}")

        ax[1,1].set_title("Imaginary part of S11")
        ax[1,1].set_xlabel("Angular Frequency (Hz)")
        ax[1,1].set_ylabel("Im{S11}")
        ax[1,1].grid(True, which='both')
    else:
        ax[0].plot(freq, 10*np.log10(np.abs(S_11_data_corr_rel)), color = color)
        ax[1].plot(freq, np.angle(S_11_data_corr_rel), color = color)

        ax[0].set_title("Reflection Magnitude |S11|")
        ax[0].set_xlabel("Angular Frequency (Hz)")
        ax[0].set_ylabel("|S11|")

        ax[1].set_title("Reflection Phase ∠S11")
        ax[1].set_xlabel("Angular Frequency (Hz)")
        ax[1].set_ylabel("Phase (radians)")
        ax[1].grid(True, which='both')

ax[0].legend(legend, title = stepping)
fig.suptitle(f"Measurement: {meas}")
fig.subplots_adjust(0.1, 0.11, 0.95, 0.88, 0.25, 0.2)

# parent_folder_path = "/home/kth-user/Documents/presentation_material/"
# if save_fig:
#     file_name = parent_folder_path + fab_name + "/" + test_name + "/" + test_name + "_" + png_title + ".png"
#     fig.savefig(file_name)
#     print(f"Saving PNG: {parent_folder_path + fab_name + "/" + test_name + "/" + test_name + "_" + png_title + ".png"}")

plt.show()
