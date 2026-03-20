import numpy as np
from tqdm import tqdm

from presto import test
from presto import lockin, utils
from presto.hardware import AdcMode, DacMode, AdcFSample, DacFSample

# Network settings for Presto Hardware
ADDRESS = '130.237.35.90'       # IP Address
PORT    = 42873                 # TCP Port

# Input (ADC) settings
INPUT_PORT = 5                  # Correlated vacuum input to presto, output frm JPA
ADC_ATT = 0.0                   # dB, 0.0 to 27.0
INPUT_NCO = 0                   # Hz, 0 to 10 GHz
DF = 10e3                      # MHz

# FLUX PUMP Output (DAC) settings
FLUX_PORT = 2                   # Pump frequency comb output from presto, input to JPA
PUMP_AMP = 0.0                # amplitude of pump signal, 0 for vacuum
PHASEI = 0.0                    # rad
PHASEQ = PHASEI - np.pi / 2     # rad
f0= 4.428e9                     # Resonance Frequency (Hz)  4427780358
PUMP_NCO = 8.84e9               # NCO frequency for pump set to 8.4 GHz
PUMP_FREQ = 2 * f0 - PUMP_NCO   # Hz, 0 to 500 MHz, intermediate frequency

# DC BIAS settings
DC_PORT = 2                     # DC Bias for optimal operating point of JPA   
DAC_CURR = 32_000               # μA, 2250 to 40500   
DC_BIAS = 2.2                   # Set LKIPA Resonance to 4.428 GHz, taken from latest calibration (2.2 for PUMP OFF, 0.5 for PUMP= 0.25)

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
    n_pix: int
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

    print('MEASUREMENT PARAMETERS:')
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
    
    return data_all, dt, fs, nr_samples

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
                I_all[pix]= I_all[pix] #- np.mean(I_all[pix])  # remove DC component