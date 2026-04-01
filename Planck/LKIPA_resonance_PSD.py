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

# Network settings for Presto Hardware
ADDRESS = '192.168.88.53'       # IP Address
PORT    = None                 # TCP Port

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
DC_BIAS = 0                     # Set LKIPA Resonance to 4.428 GHz, taken from latest calibration (2.2 for PUMP OFF, 0.5 for PUMP= 0.25)

# Converter configuration for Presto hardware
CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Mixed,
    "adc_fsample": AdcFSample.G2,
    "dac_mode": DacMode.Mixed04,
    "dac_fsample": DacFSample.G8,
}     

# Number of pixels to be captured
N_PIX = 1_000 

# Define data acquisition function
def data_acquisition(
    address: str,
    port: int,
    converter_configuration: dict,
    input_port: int,
    adc_att: float,
    input_nco: float,
    output_port: int,
    dac_curr: int,
    amp: float,
    freq: float,
    phasei: float,
    phaseq: float,
    output_nco: float,
    df: float,
    dcb_port: int,
    dcb_amp: float,
    n_pix: int,
):
    with test.Test(address=address, port=port, **converter_configuration) as tst:
        # Get extra samples at the beginning and throw them away
        extra = 1000
        # Calculate number of samples from DF
        nr_samples = int(round(tst.get_fs("adc") / df))

        # Configure mixers for input and output ports
        tst.hardware.configure_mixer(input_nco, in_ports=input_port, sync=False)
        tst.hardware.configure_mixer(output_nco, out_ports=output_port, sync=True)

        # Configure ADC and DAC settings
        tst.hardware.set_adc_attenuation(input_port, adc_att)
        tst.hardware.set_dac_current(output_port, dac_curr)
        
        # Set DC bias for 4.2GHz operating point of JPA
        tst.hardware.set_dc_bias(port=dcb_port, bias=dcb_amp)
        tst.hardware.sleep(1e-4)

        # Configure output signal for pump tone
        tst.set_frequency(output_port, freq)
        tst.set_phase(output_port, phasei, phaseq)
        tst.set_scale(output_port, scale_i = amp, scale_q = amp)
        tst.hardware.sleep(1e-4)

        # Print hardware configuration statement
        print('Hardware configuration successful, initiating data acquisition ...')

        myrun = time.strftime("%Y-%m-%d_%H_%M_%S")  

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

        # set all outputs to 0
        tst.hardware.set_dc_bias(port=dcb_port, bias=0.0)
        tst.set_scale(output_port, scale_i = 0, scale_q = 0)
        
        # convert data list to np.array
        data_all = np.array(data_all)

        # Measurement metadata
        dt = tst.get_dt("adc")*1e9
        fs = tst.get_fs("adc")*1e-9

        # Print completion statement
        print('Data Acquisition Complete.')

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

    # Data points captured per pixel
    N_datastream = np.shape(data_all)[1]
    print(f"Data points captured per pixel: {N_datastream}")

    # Number of samples per pixel 
    print(f"Number of samples per pixel: {nr_samples}")
    
    return data_all, dt, fs, nr_samples, myrun

def remove_DC(
        data_all,
        converter_configuration=CONVERTER_CONFIGURATION,
        n_pix=N_PIX,
):
    if converter_configuration["adc_mode"] == AdcMode.Mixed:
        print("Data format: Mixed mode (I and Q interleaved)")

        # Convert raw ADC data to full-scale (FS) units and separate I and Q components
        I_all = data_all[:, 0::2]
        Q_all = data_all[:, 1::2]   # left alone for now !!!
        print(f"Shape of I data: {I_all.shape}")

        # Assign data array for raw pixel I data
        for pix in range(n_pix):
            I_all[pix]= I_all[pix] - np.mean(I_all[pix])  # remove DC component
    
    elif converter_configuration["adc_mode"] == AdcMode.Direct:
        print("Data format: Direct mode (I only)")

        # Convert raw ADC data to full-scale (FS) units
        I_all = data_all
        print(f"Shape of I data: {I_all.shape}")

        for pix in range(N_PIX):
            I_all[pix]= I_all[pix] - np.mean(I_all[pix])  # remove DC component

    return I_all 


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
):
    A_bg, B_bg, A_peak, f_0, gamma = cf(
        lorentzian_fit_func,
        f_arr_bandwidth,
        PSD_bandwidth,
        p0=[0.5, 0.5, 0.16, 0.428, 0.5e-3]
    )[0]

    print('\n=======================')
    print('FITTING PARAMETERS:')
    print('A_background = ', str(np.round(A_bg, 2)))
    print('B_background = ', str(np.round(B_bg, 2)))
    print('A_peak = ', str(np.round(A_peak, 2)))
    print('f0 = ', str(np.round(f_0 + 4, 5)), 'GHz')
    print('gamma = ', str(np.round(np.abs(gamma * 1e3), 3)), 'MHz')

    return A_bg, B_bg, A_peak, f_0, gamma

def plot_PSD_bw(
        PSD_bandwidth,
        f_arr_bandwidth,
        fit_params,
        temp,
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

    # temperature string
    temp_str = str(temp)
    
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
        color='darkorange'
        )
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Magnitude [a.u.]")
    ax.set_title("Power Spectral Density: Temperature = " + temp_str+ 'mK', fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

    # Save plot
    fig.savefig(
        fname=f'D:/Planck Spectroscopy 2026-03/LKIPA Resonance/Plots/LKIPA_resonance_PSD-temp={temp_str}.png',
        dpi=200,
    )

    plt.close(fig)

# Save data function
def save_data(
        folder, 
        file, 
        sample, 
        myrun, 
        df, 
        dt, 
        n_pix, 
        n_samples, 
        data_all, 
        temp,
        fit_params,
        ):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(folder, file), "a") as savefile:
        # String as handles
        sample_str = '{}/sample'.format(myrun)
        df_data_str = "{}/df".format(myrun)
        time_series_str = "{}/Time Series Data".format(myrun)
        dt_str = '{}/dt'.format(myrun)
        npix_data_str = "{}/Pixels".format(myrun)
        nsamples_series_str = "{}/Samples per pixel".format(myrun)
        temp_str = "{}/Temperature".format(myrun)
        fit_params_str = "{}/Fit parameters".format(myrun)
        
        # Write data to datasets
        dt_string = h5py.string_dtype(encoding="utf-8")
        savefile.create_dataset(sample_str, data=str(sample), dtype=dt_string)
        
        savefile.create_dataset(df_data_str, (np.shape(df)),
                                dtype=float, data=df)
        
        data_all_arr = np.asarray(data_all)
        savefile.create_dataset(time_series_str, data=data_all_arr)
        
        savefile.create_dataset(dt_str, (np.shape(dt)),
                                dtype=float, data=dt)
        
        savefile.create_dataset(npix_data_str, (np.shape(n_pix)),
                                dtype=float, data=n_pix)
        
        savefile.create_dataset(nsamples_series_str , (np.shape(n_samples)),
                                dtype=complex, data=n_samples)
        
        savefile.create_dataset(temp_str, (np.shape(temp)),
                                dtype=complex, data=temp)
        
        savefile.create_dataset(fit_params_str, (np.shape(fit_params)),
                                dtype=complex, data=fit_params)
        
        # Write dataset attributes
        savefile[df_data_str].attrs["Unit"] = "Hz"
        savefile[dt_str].attrs["Unit"] = "ns"
        savefile[time_series_str].attrs["Unit"] = "fsu complex"
        savefile[temp_str].attrs["Unit"] = "mK"


def get_lkipa_resonance(
    address: str,
    port: int,
    converter_configuration: dict,
    input_port: int,
    adc_att: float,
    input_nco: float,
    output_port: int,
    dac_curr: int,
    amp: float,
    freq: float,
    phasei: float,
    phaseq: float,
    output_nco: float,
    df: float,
    dcb_port: int,
    dcb_amp: float,
    n_pix: int,
    save_folder,
    save_file,
    temp,
    f_L = 0.42,
    f_R = 0.44,
):
    # Run data acquisition
    data_all, dt, fs, n_samples, myrun = data_acquisition(
        address=address,
        port=port,
        converter_configuration=converter_configuration,
        input_port=input_port,
        adc_att=adc_att,
        input_nco=input_nco,
        output_port=output_port,
        dac_curr=dac_curr,
        amp=amp,
        freq=freq,
        phasei=phasei,
        phaseq=phaseq,
        output_nco=output_nco,
        df=df,
        dcb_port=dcb_port,
        dcb_amp=dcb_amp,
        n_pix=n_pix,
    )

    # Get untwisted dc removed time series
    I_all = remove_DC(data_all=data_all)

    # GET POWER SPECTRAL DENSITY
    PSD_avg, f_arr, t_arr = get_PSD_avg(
        I_all=I_all,
        n_samples=n_samples,
        dt=dt
    )

    # SELECT BANDWIDTH
    PSD_bw, f_bw = get_PSD_bw(
        PSD_avg=PSD_avg,
        f_arr=f_arr,
        f_L=f_L,
        f_R=f_R,
    )

    # Get fitting parameters for PSD
    fit_params = lorentz_fit(
        PSD_bandwidth=PSD_bw,
        f_arr_bandwidth=f_bw,
        lorentzian_fit_func=lorentzian_fit_func,
    )

    # Save plot
    plot_PSD_bw(
        PSD_bandwidth=PSD_bw,
        f_arr_bandwidth=f_bw,
        fit_params=fit_params,
        temp=temp,
)
    print('\n=======================')
    print('Plot Saved')

    # Save data
    save_data(
        folder=save_folder,
        file=save_file,
        sample='LKIPA',
        myrun=myrun,
        df=df,
        dt=dt,
        n_pix=n_pix,
        n_samples=n_samples,
        data_all=data_all,
        temp=temp,
        fit_params=fit_params,
    )
    print('\n=======================')
    print('Data Saved')

############################ TESTING #############################################

# save_folder='D:/Planck Spectroscopy 2026-03/LKIPA Resonance'
# myrun       = time.strftime("%Y-%m-%d_%H_%M_%S")                # Save experimental run for each timestamp
# save_file   = r"{}.hdf5".format(myrun)                          # Save data in hdf5 file for current run

# get_lkipa_resonance(
#     address=ADDRESS,
#         port=PORT,
#         converter_configuration=CONVERTER_CONFIGURATION,
#         input_port=INPUT_PORT,
#         adc_att=ADC_ATT,
#         input_nco=INPUT_NCO,
#         output_port=FLUX_PORT,
#         dac_curr=DAC_CURR,
#         amp=PUMP_AMP,
#         freq=PUMP_FREQ,
#         phasei=PHASEI,
#         phaseq=PHASEQ,
#         output_nco=PUMP_NCO,
#         df=DF,
#         dcb_port=DC_PORT,
#         dcb_amp=DC_BIAS,
#         n_pix=N_PIX,
#         f_L=0.42,
#         f_R=0.44,
#         save_folder=save_folder,
#         save_file=save_file,
#         temp = 10
# )










