import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from presto import test
from presto.hardware import AdcMode, DacMode, AdcFSample, DacFSample

# Network settings for Presto Hardware
ADDRESS = '130.237.35.90'   # IP Address
PORT    = 42873             # TCP Port

# input (ADC) settings
INPUT_PORT = 9          # Dummy loopback port
ADC_ATT = 0.0           # dB, 0.0 to 27.0
INPUT_NCO = 1.5e9       # Hz, 0 to 10 GHz
DF = 100e3                # Hz

# output (DAC) settings
OUTPUT_PORT = 9         # Dummy loopback port
DAC_CURR = 32_000       # μA, 2250 to 40500
AMP = 0.0               # FS, 0.0 to 1.0
FREQ = 24e6             # Hz, 0 to 500 MHz, intermediate frequency
PHASEI = 0.0            # rad
PHASEQ = PHASEI - np.pi / 2  # rad
OUTPUT_NCO = 1.5e9      # Hz, 0 to 10 GHz

CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Direct,
    "adc_fsample": AdcFSample.G2,
    "dac_mode": DacMode.Mixed02,
    "dac_fsample": DacFSample.G6,
}  


with test.Test(address=ADDRESS, port=PORT, **CONVERTER_CONFIGURATION) as tst:
    # get extra samples at the beginning and throw them away
    extra = 250
    # calculate number of samples from DF
    nr_samples = int(round(tst.get_fs("adc") / DF))

    # NOTE:
    # in Mixed mode, maximum number for `extra + nr_samples` is 2**29 (512 MiS)
    # corresponding to ≈0.5 s of data, or ≈2 Hz resolution

    tst.hardware.configure_mixer(INPUT_NCO, in_ports=INPUT_PORT, sync=False)
    tst.hardware.configure_mixer(OUTPUT_NCO, out_ports=OUTPUT_PORT, sync=True)
    tst.hardware.set_adc_attenuation(INPUT_PORT, ADC_ATT)
    tst.hardware.set_dac_current(OUTPUT_PORT, DAC_CURR)
    tst.hardware.sleep(0.1)

    tst.set_frequency(OUTPUT_PORT, FREQ)
    tst.set_phase(OUTPUT_PORT, PHASEI, PHASEQ)
    tst.set_scale(OUTPUT_PORT, scale_i = AMP, scale_q = 0)

    tst.hardware.set_run(False)
    tst.set_dma_source(INPUT_PORT)
    tst.start_dma(extra + nr_samples)
    tst.hardware.set_run(True)
    tst.wait_for_dma()
    tst.stop_dma()
    data = tst.get_dma_data(extra + nr_samples)
    tst.hardware.check_adc_intr_status()

# throw away initial `extra` data points
data = data[-nr_samples:]

# check which samples are close to saturation
idx_sat = np.logical_or(data >= 32764, data <= -32764)
nr_sat = np.sum(idx_sat)
print(f"Found {nr_sat:d} data points at 100% input range")

# convert data to full-scale (FS) units
data = data / 32767

# make time and frequency arrays
t_arr = tst.get_dt("adc") * np.arange(nr_samples)
f_arr = np.fft.rfftfreq(nr_samples, tst.get_dt("adc"))

print('Timestamps := ', t_arr)
print(
    'Effective Sampling Frequency := ', 
    np.round(1e-9 / tst.get_dt("adc"), 2), 
    ' GHz '
    )
print('no. of samples := ', nr_samples)

# FFT
data_fft = np.fft.rfft(data) / nr_samples

# plot
fig1, ax1 = plt.subplots(
    nrows=1,                    # set to 2 for time and frequency domain plots
    ncols=1, 
    constrained_layout=True
    )

# ax11, ax12 = ax1
ax1.plot(1e6 * t_arr, data)
ax1.plot(1e6 * t_arr[idx_sat], data[idx_sat], ".", ms=9)
#ax12.semilogy(1e-6 * f_arr, np.abs(data_fft))
ax1.set_xlabel("Time [μs]")
#ax12.set_xlabel("Frequency [MHz]")
ax1.grid()
#ax12.grid()

if hasattr(sys, "ps1"):
    fig1.show()
else:
    plt.show()
