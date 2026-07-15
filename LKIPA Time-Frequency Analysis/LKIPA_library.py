from turtle import clear

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from presto import test
from presto import lockin, utils
from presto.hardware import AdcMode, DacMode, AdcFSample, DacFSample
from scipy.optimize import curve_fit as cf
import time
import os
import h5py
from IPython.display import display, clear_output

# Network settings for Presto Hardware
# ADDRESS = '192.168.88.53'       # IP Address
# PORT    = None                 # TCP Port
ADDRESS = '130.237.35.90'       # IP Address
PORT    = 42873                 # TCP Port

# Input (ADC) settings
INPUT_PORT = 5                  # Correlated vacuum input to presto, output frm JPA
ADC_ATT = 0.0                   # dB, 0.0 to 27.0
INPUT_NCO = 0                   # Hz, 0 to 10 GHz
DF = 10e3                       # MHz

# FLUX PUMP Output (DAC) settings
FLUX_PORT = 2                   # Pump frequency comb output from presto, input to JPA
PUMP_AMP = 0.0                  # amplitude of pump signal, 0 for vacuum
PHASEI = 0.0                    # rad
PHASEQ = PHASEI - np.pi / 2     # rad
f0= 4.428e9                     # Resonance Frequency (Hz)  4427780358
PUMP_NCO = 8.4e9                # NCO frequency for pump set to 8.4 GHz
PUMP_FREQ = 2 * f0 - PUMP_NCO   # Hz, 0 to 500 MHz, intermediate frequency

# DC BIAS settings
DC_PORT = 2                     # DC Bias for optimal operating point of JPA   
DAC_CURR = 32_000               # μA, 2250 to 40500   
DC_BIAS = 1.7                    # Set LKIPA Resonance to 4.428 GHz, taken from latest calibration (2.2 for PUMP OFF, 0.5 for PUMP= 0.25)

# Converter configuration for Presto hardware
CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Mixed,
    "adc_fsample": AdcFSample.G2,
    "dac_mode": DacMode.Mixed04,
    "dac_fsample": DacFSample.G8,
}     

# Number of pixels to be captured
N_PIX = 1_000 


def remove_DC(
        data_all,
        n_pix,
        converter_configuration=CONVERTER_CONFIGURATION,
        verbose=False,
):
    if converter_configuration["adc_mode"] == AdcMode.Mixed:
        if verbose: print("Data format: Mixed mode (I and Q interleaved)")

        # Convert raw ADC data to full-scale (FS) units and separate I and Q components
        I_all = data_all[:, 0::2]
        Q_all = data_all[:, 1::2]   
        if verbose: print(f"Shape of I data: {I_all.shape}")

        # Assign data array for raw pixel I & Q  data
        for pix in range(n_pix):
            I_all[pix]= I_all[pix] - np.mean(I_all[pix])  # remove DC component
            Q_all[pix]= Q_all[pix] - np.mean(Q_all[pix])  # remove DC component
        return I_all, Q_all 
    
    elif converter_configuration["adc_mode"] == AdcMode.Direct:
        if verbose: print("Data format: Direct mode (I only)")

        # Convert raw ADC data to full-scale (FS) units
        I_all = data_all
        if verbose: print(f"Shape of I data: {I_all.shape}")

        for pix in range(n_pix):
            I_all[pix]= I_all[pix] - np.mean(I_all[pix])  # remove DC component

        return I_all
    

# Save data to hdf5 file 

def save_data(folder, file, myrun, pump_freq, pump_idx_dict, df, f_NCO, dc_bias, N_pixels, I_arr, Q_arr, t_arr, f_s = 1e9):

    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(folder, file), "a") as savefile:

        pump_idx = pump_idx_dict["pump_idx"]
        pump_amp = pump_idx_dict["pump_amp"]

        # String as handles
        pump_freq_str = "{}/{}/pump freq".format(myrun, pump_idx)
        pump_amp_str = "{}/{}/pump amp".format(myrun, pump_idx)
        df_data_str = "{}/{}/df".format(myrun, pump_idx)
        f_NCO_str = "{}/{}/f_NCO".format(myrun, pump_idx)
        dc_bias_str = "{}/{}/dc bias".format(myrun, pump_idx)
        N_pixels_str = "{}/{}/N pixels".format(myrun, pump_idx)
        I_arr_str = "{}/{}/I_arr".format(myrun, pump_idx)
        Q_arr_str = "{}/{}/Q_arr".format(myrun, pump_idx)
        t_arr_str = "{}/{}/t_arr".format(myrun, pump_idx)
        fs_str = "{}/{}/fs".format(myrun, pump_idx)

        # Write data to datasets
        savefile.create_dataset(pump_freq_str, (np.shape(pump_freq)),
                                dtype=float, data=pump_freq)
        savefile.create_dataset(pump_amp_str, (np.shape(pump_amp)),
                                dtype=float, data=pump_amp)
        savefile.create_dataset(df_data_str, (np.shape(df)),
                                dtype=float, data=df)
        savefile.create_dataset(f_NCO_str, (np.shape(f_NCO)),
                                dtype=float, data=f_NCO)
        savefile.create_dataset(dc_bias_str, (np.shape(dc_bias)),
                                dtype=float, data=dc_bias)
        savefile.create_dataset(N_pixels_str, (np.shape(N_pixels)),
                                dtype=int, data=N_pixels)
        savefile.create_dataset(I_arr_str, (np.shape(I_arr)),
                                dtype=float, data=I_arr)
        savefile.create_dataset(Q_arr_str, (np.shape(Q_arr)),
                                dtype=float, data=Q_arr)
        savefile.create_dataset(t_arr_str, (np.shape(t_arr)),
                                dtype=float, data=t_arr)
        savefile.create_dataset(fs_str, (np.shape(f_s)),
                                dtype=float, data=f_s)

        # Write dataset attributes
        savefile[pump_freq_str].attrs["Unit"] = "Hz"
        savefile[pump_amp_str].attrs["Unit"] = "fsu"
        savefile[f_NCO_str].attrs["Unit"] = "Hz"
        savefile[dc_bias_str].attrs["Unit"] = "V"
        savefile[df_data_str].attrs["Unit"] = "Hz"
        savefile[Q_arr_str].attrs["Unit"] = "fsu"
        savefile[t_arr_str].attrs["Unit"] = "μs"
        savefile[fs_str].attrs["Unit"] = "Hz"

##########################################################################
######### Time Series data acquisition routine for Presto hardware #######
##########################################################################

def data_acquisition(
    address: str,
    port: int,
    converter_configuration: dict,
    input_port: int,
    adc_att: float,
    input_nco: float,
    output_port: int,
    dac_curr: int,
    amp_list: list,
    freq: float,
    phasei: float,
    phaseq: float,
    output_nco: float,
    df: float,
    dcb_port: int,
    dcb_amp: float,
    n_pix: int,
    myrun: str,
    folder: str,
    file: str
):
    with test.Test(address=address, port=port, **converter_configuration) as tst:
        # Get extra samples at the beginning and throw them away
        extra = 1000
        # Calculate number of samples from DF
        nr_samples = int(round(tst.get_fs("adc") / df))

        # Measurement metadata
        dt = tst.get_dt("adc")*1e9                  # ns
        fs = tst.get_fs("adc")*1e-9                 # GHz
        t_arr = dt * np.arange(0, nr_samples, 1)
    

        # Configure mixers for input and output ports
        tst.hardware.configure_mixer(input_nco, in_ports=input_port, sync=False)
        tst.hardware.configure_mixer(output_nco, out_ports=output_port, sync=True)

        # Configure ADC and DAC settings
        tst.hardware.set_adc_attenuation(input_port, adc_att)
        tst.hardware.set_dac_current(output_port, dac_curr)
        
        # Set DC bias for 4.2GHz operating point of JPA
        tst.hardware.set_dc_bias(port=dcb_port, bias=dcb_amp)
        tst.hardware.sleep(1e-4)


        for amp_idx, amp_val in enumerate(amp_list):

            # Print processing current pump amplitude statement
            print(f'Processing pump amplitude {amp_idx + 1} / {len(amp_list)} : {amp_val}')

            # Configure output signal for pump tone
            tst.set_frequency(output_port, freq)
            tst.set_phase(output_port, phasei, phaseq)
            tst.set_scale(output_port, scale_i = amp_val, scale_q = amp_val)
            tst.hardware.sleep(1e-4)

            # Print hardware configuration statement
            print('Hardware configuration successful, initiating data acquisition ...')             

            # Start data acquisition
            data_all = []
            with tqdm(total=n_pix, ncols=80) as pbar:
                for i in range(n_pix):
                    tst.hardware.set_run(False)
                    tst.set_dma_source(input_port)
                    tst.start_dma(extra + nr_samples)
                    tst.hardware.set_run(True)
                    tst.wait_for_dma()
                    tst.stop_dma()
                    data = tst.get_dma_data(extra + nr_samples)
                    tst.hardware.check_adc_intr_status()
                    
                    # Throw away initial `extra` data points, convert to FS
                    if converter_configuration["adc_mode"] == AdcMode.Mixed:
                        data = data[2*extra:] / 32767

                    elif converter_configuration["adc_mode"] == AdcMode.Direct:
                        data = data[extra:] / 32767

                    # Append to data_all list
                    data_all.append(data)
                    
                    # Update progress bar
                    pbar.update(1)
            
            # Convert data list to np.array
            data_all = np.array(data_all)

            # Remove DC component from data
            I_all, Q_all = remove_DC(data_all, n_pix, converter_configuration)

            # Save data to hdf5 file
            pump_idx_dict = {
                "pump_idx": amp_idx,
                "pump_amp": amp_val
            }

            save_data(folder, file, myrun, freq, pump_idx_dict, df, input_nco, dcb_amp, n_pix, I_all, Q_all, t_arr)

            # Print message for saving data for current pump power
            print(f"Data for pump amplitude {amp_idx + 1} / {len(amp_list)} ({amp_val}) saved successfully.")

            # clear outputs
            clear_output(wait=True)

        # set all outputs to 0
        tst.hardware.set_dc_bias(port=dcb_port, bias=0.0)
        tst.set_scale(output_port, scale_i = 0, scale_q = 0)
        

        # Print completion statement
        print('Presto outputs reset to 0, data successfully save to {}.'.format(file))

    # Print measurement metadata
    # ==========================

    print('\nMEASUREMENT PARAMETERS:')
    print('=======================')

    # Analog-to-Digital Converter Mode
    print(f'Mode: {converter_configuration["adc_mode"]}')

    # Number of pixels captured
    print(f"Number of pixels: {n_pix}")

    # Pixel duration
    print(f"Pixel time resolution (dt): {dt:.2f} ns")

    # Sampling frequency of the ADC (GHz)
    print(f"Sampling frequency (fs): {fs:.2f} GHz")

    t_meas = 1e6 / df # Measurement time per pixel (μs)
    print(f"Total measurement time: {t_meas:.1f} µs")

    # Frequency resolution (kHz)
    print(f"Frequency resolution (DF): {df/1e3:.1f} kHz")


# Data retrieval function from hdf5 file
def retrieve_data(folder, file, myrun, pump_idx):
    with h5py.File(os.path.join(folder, file), "r") as savefile:
        pump_freq_str = "{}/{}/pump freq".format(myrun, pump_idx)
        pump_amp_str = "{}/{}/pump amp".format(myrun, pump_idx)
        df_data_str = "{}/{}/df".format(myrun, pump_idx)
        f_NCO_str = "{}/{}/f_NCO".format(myrun, pump_idx)
        dc_bias_str = "{}/{}/dc bias".format(myrun, pump_idx)
        N_pixels_str = "{}/{}/N pixels".format(myrun, pump_idx)
        I_arr_str = "{}/{}/I_arr".format(myrun, pump_idx)
        Q_arr_str = "{}/{}/Q_arr".format(myrun, pump_idx)
        t_arr_str = "{}/{}/t_arr".format(myrun, pump_idx)
        fs_str = "{}/{}/fs".format(myrun, pump_idx)

        # Retrieve data from datasets
        pump_freq = savefile[pump_freq_str][()]
        pump_amp = savefile[pump_amp_str][()]
        df_data = savefile[df_data_str][()]
        f_NCO = savefile[f_NCO_str][()]
        dc_bias = savefile[dc_bias_str][()]
        N_pixels = savefile[N_pixels_str][()]
        I_arr = savefile[I_arr_str][()]
        Q_arr = savefile[Q_arr_str][()]
        t_arr = savefile[t_arr_str][()]
        fs_data = savefile[fs_str][()]

    return (pump_freq, pump_amp, df_data, f_NCO, dc_bias, N_pixels,
            I_arr, Q_arr, t_arr, fs_data)

####################################
##### Correlation Functions ########
####################################


def get_correlation(I_all_list, Q_all_list):

    n_samples = np.shape(I_all_list)[1]
    n_pix = np.shape(I_all_list)[0]

    auto_xx_list = np.zeros((n_pix, n_samples))
    auto_pp_list = np.zeros((n_pix, n_samples))
    cross_xp_list = np.zeros((n_pix, n_samples))

    with tqdm(total= n_pix, ncols = 80) as pbar:

        for pix in range(n_pix):
            
            auto_xx_list[pix] = np.correlate(
                a=I_all_list[pix], 
                v=I_all_list[pix],
                mode='same'
            )

            auto_pp_list[pix] = np.correlate(
                    a=Q_all_list[pix], 
                    v=Q_all_list[pix],
                    mode='same'
                    )

            cross_xp_list[pix] = np.correlate(
                    a=I_all_list[pix],
                    v=Q_all_list[pix],
                    mode='same'
            )
            pbar.update(1)

    # Pixel averaged correlation functions
    auto_xx = np.mean(auto_xx_list, axis=0)
    auto_pp = np.mean(auto_pp_list, axis=0)
    cross_xp = np.mean(cross_xp_list, axis=0)

    return auto_xx, auto_pp, cross_xp
                    

def origin_removed(function):

    n_samples = len(function)

    function_origin_removed = np.concatenate([function[:n_samples//2], function[n_samples//2+1:]])

    return function_origin_removed














###### PSD Analysis Functions #####

def get_PSD_avg(
    I_all,
    n_samples,
    dt,
    n_pix = N_PIX    
):
    t_arr = dt * np.arange(0, n_samples, 1) * 1e-3    # μs, convert from ns seconds

    # FFT
    # ===

    # frequency array for FFT
    f_arr = np.fft.rfftfreq(n_samples, dt)  # Hz

    fft_data_list = [] # np.zeros((N_PIX, n_samples))
    for pix in range(n_pix):
        fft_data_list.append(np.fft.rfft(I_all[pix]))

    # FFT of all time series
    fft_data_list = np.array(fft_data_list)

    # PSD 
    # === 
    PSD_avg = np.mean(np.abs(fft_data_list)  ** 2, axis = 0)   # PSD averaged over all pixels

    # Untwist
    f_arr = np.concatenate(
        (
            f_arr[n_samples // 2:], 
            f_arr[0: n_samples // 2]
    )
    )

    PSD_avg = np.concatenate(
        (
            PSD_avg[n_samples // 2:], 
            PSD_avg[0: n_samples // 2]
    )
    )

    return PSD_avg, f_arr, t_arr

def get_PSD_bw(
        PSD_avg,
        f_arr,
        f_L,
        f_R
        ):
    
    L_idx = np.argmin(np.abs(f_arr - f_L))
    R_idx = np.argmin(np.abs(f_arr - f_R))

    f_arr_bandwidth = f_arr[L_idx: R_idx+1]
    PSD_bandwidth = PSD_avg[L_idx: R_idx+1]

    return PSD_bandwidth, f_arr_bandwidth

def lorentzian_fit_func(
        f,
        A_bg,
        B_bg,
        A_peak,
        f_0,
        gamma
):
    lorentzian = A_bg * f + B_bg  + A_peak / (((f - f_0)/gamma) ** 2 + 1)
    return lorentzian

def lorentz_fit(
        PSD_bandwidth,
        f_arr_bandwidth,
        lorentzian_fit_func,
        verbose=False,
):
    A_bg, B_bg, A_peak, f_0, gamma = cf(
        lorentzian_fit_func,
        f_arr_bandwidth,
        PSD_bandwidth,
        p0=[0.5, 0.5, 0.16, 0.428, 0.5e-3]
    )[0]

    if verbose:

        print('\n=======================')
        print('FITTING PARAMETERS:')
        print('A_background = ', str(np.round(A_bg, 2)))
        print('B_background = ', str(np.round(B_bg, 2)))
        print('A_peak = ', str(np.round(A_peak, 4)))
        print('f0 = ', str(np.round(f_0 + 4, 5)), 'GHz')
        print('gamma = ', str(np.round(np.abs(gamma * 1e3), 3)), 'MHz')

    return A_bg, B_bg, A_peak, f_0, gamma

def plot_PSD_bw(
        PSD_bandwidth,
        f_arr_bandwidth,
        fit_params,
):
    
    # get fitting function
    fit_func = lorentzian_fit_func(
        f_arr_bandwidth, 
        A_bg=fit_params[0],
        B_bg=fit_params[1],
        A_peak=fit_params[2],
        f_0=fit_params[3],
        gamma=fit_params[4]
    )
    
    # PLOT
    # ====
    fig, ax = plt.subplots(
        constrained_layout=True, 
        figsize=(8, 6)
        )

    # Plot PSD
    ax.grid(alpha=0.8, linestyle='--')
    ax.plot(4 + f_arr_bandwidth, PSD_bandwidth, label="$\\langle \\tilde V^2[\\omega] \\rangle$", color = "b", lw=1.3)
    ax.plot(
        4 + f_arr_bandwidth, 
        fit_func,
        label='Fit',
        lw=2,
        ls= '--',
        color='darkorange'
        )
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Magnitude [a.u.]")
    ax.set_title("Power Spectral Density", fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

    # Save plot
    # fig.savefig(
    #     fname=f'D:/Planck Spectroscopy 2026-03/LKIPA Resonance/Plots/LKIPA_resonance_PSD-temp={temp_str}.png',
    #     dpi=200,
    # )

    #plt.close(fig)


