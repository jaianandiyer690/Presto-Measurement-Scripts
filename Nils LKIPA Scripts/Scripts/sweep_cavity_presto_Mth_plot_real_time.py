import h5py
import os
import platform
import inspect
import numpy as np
import sys
import time

import threading
import queue

# Platform-specific imports
if sys.platform.startswith('win'):
    import msvcrt
else:
    import termios
    import tty
    import select

# Platform-independent getch and kbhit
def getch():
    if sys.platform.startswith('win'):
        return msvcrt.getch().decode('utf-8')
    else:
        return sys.stdin.read(1)

def kbhit():
    if sys.platform.startswith('win'):
        return msvcrt.kbhit()
    else:
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return dr != []

def control():
    key=""
    print("Press Q to quit, press u to update measurement values")

    if not sys.platform.startswith('win'):
        # Linux/macOS: Set terminal to cbreak mode
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        while not event.is_set():
            if kbhit():
                key = getch()
                if key == 'Q':
                    print("\nQ detected. Exiting control thread.")
                    event.set()
                if key == 'u':                    
                    event2.set()
                    key=""
            time.sleep(0.05)
    finally:
        if not sys.platform.startswith('win'):
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
# from functools import partial
import json

from scipy.optimize import curve_fit

import presto
from presto import lockin, utils

from parameter_file import *


def meas():
    f_nco = 3.9 *1e9  # dummy variable

    f_start = 4.2* 1e9
    f_stop = 4.4* 1e9
     
    _df = 300_000
    f_delta = 300_000
    nr_pxs = 100
    

    v_amp=0.01
    vdc=0
    v_pump = 0.0
    
    f_center = 4400913568.342668
    f_nco_pump = 2*f_center

    f_arr = np.arange(f_start, f_stop + f_delta, f_delta)
    nr_freq = len(f_arr)
    HSB_arr=np.zeros(len(f_arr),dtype=np.complex128)  
    
    # y_max=10*v_amp
    # y_max=0.0005*v_amp
    # tau=-7.55e-08
    # tau=-1.105e-7
    tau = -1.095e-7
    phi0=-2
    with lockin.Lockin(address=BOX_ADDRESS, **CONVERTER_RATES, port = PORT) as lck:

        lck.hardware.configure_mixer(
            freq=f_nco,
            in_ports=v_input_port,
            out_ports=v_output_signal_port
        )
        print("NCO set")

        outgroup = lck.add_output_group(v_output_signal_port, 1)
        outgroup.set_phases(phases=0.0, phases_q=-np.pi / 2)
        lck.hardware.set_dc_bias(0.0, vdc_port)

        # Create a lockin output group object, for one frequency, upper sideband only
        outgroup_pump = lck.add_output_group(v_output_pump_port, 1)
        outgroup_pump.set_phases(phases=0.0, phases_q=-np.pi / 2)

        ingroup = lck.add_input_group(v_input_port, 1)
        ingroup.set_phases(0.0)
        lck.apply_settings()
        time.sleep(0.1)

        fs = lck.get_fs("dac")
        print(f"Sampling frequency fs: {fs:.3e} Hz")
        nr_samples = int(round(fs / _df))
        df = fs / nr_samples
        lck.set_df(df)
        outgroup.set_amplitudes(v_amp)
        outgroup.set_frequencies(0)
        ingroup.set_frequencies(0)
        
        lck.apply_settings()
        time.sleep(0.1)

        # Verify
        print(f"Applied df: {lck.get_df()} Hz")

        # resp_arr = np.zeros(nr_freq, dtype=np.complex128)
        counter=0
        while (event.is_set()==False):
            if event2.is_set():
                with open("values.txt") as file:
                    values=json.load(file)                
                    v_amp=np.abs(values["vamp"])
                    if v_amp>1.0:
                        print("v_amp too large, applying 1.0 instead" )                       
                    
                    f_start=values["f_start"]
                    f_stop=values["f_stop"]                    
                    _df=values["df"]
                    f_delta=values["f_delta"]                    
                    tau=values["tau"]                    
                    phi0=values["phi0"]
                    vdc=values["dcbias"]
                    v_pump=values["vpump"]
                    
                nr_samples = int(round(fs / _df))
                df = fs / nr_samples
                lck.set_df(df)   
                outgroup.set_amplitudes(v_amp)
                outgroup_pump.set_amplitudes(v_pump)
                lck.apply_settings()
                time.sleep(0.1)
                f_arr = np.arange(f_start, f_stop + f_delta, f_delta)  
                if len(f_arr)!=len(HSB_arr):                
                    counter=0
                # t_arr=np.linspace(0,2*np.pi,200)
                    HSB_arr=np.zeros(len(f_arr),dtype=np.complex128)
                lck.hardware.set_dc_bias(vdc, vdc_port)

                event2.clear() 
                print("measurement values updated")
            f=f_arr[counter]
                    # Set frequency
            lck.hardware.configure_mixer(
                freq=f,
                in_ports=v_input_port,
                out_ports=v_output_signal_port
                # tune=False
            )
            
            lck.apply_settings()
            time.sleep(0.001)

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
            HSB_arr[counter]=np.mean(HSB)*np.exp(-1j*(2*np.pi*f*tau+phi0))
            if counter<len(f_arr)-1:
                HSB_arr[counter+1]=0+0j
            datastream.put((f_arr,HSB_arr))
            counter+=1
            if counter>=len(f_arr):
                counter=0
                

        #######################################################################
        # RF off
        outgroup.set_amplitudes(0.0)  # amplitude [0.0 - 1.0]
        outgroup_pump.set_amplitudes(0.0) 
        lck.apply_settings()
        time.sleep(0.1)
        lck.hardware.set_dc_bias(0.0, 1)
        lck.hardware.set_dc_bias(0.0, vdc_port)
        lck.apply_settings()
        print('Data_gen: Done\n', flush=True)


""" setup figure"""
fig=plt.figure()
ax=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
# line1, = ax.plot(f_arr,np.zeros_like(f_arr), '-')
# line2, = ax2.plot([],[], '-')
cond=1  

def gen():
    global cond
    j=0   
    while cond:
        j+=1
        yield j
    
def initialize_plot():
    # ax2.set_xlabel("frequency (Hz)")
    # ax.set_ylabel("mag")
    # ax2.set_ylabel("phase")
    return
  
def animate(i):

    global cond
    if event.is_set():
        cond=0
        if datastream.empty()==False:
            x,y=datastream.get()
            # count,y_new=datastream.get()
            ax.clear()
            ax2.clear()
            ax.plot(x,np.abs(y))
            ax2.plot(x,np.angle(y))
            while datastream.empty()==False: #flush queue
                x,y=datastream.get()
        print("plotting interrupted")
    else:     
        while True:
            if datastream.empty()==False:
                
                x,y=datastream.get() 
                ax.clear()
                ax2.clear()
                ax.plot(x,np.abs(y))
                ax2.plot(x,np.angle(y))
                while datastream.empty()==False: #flush queue
                    x,y=datastream.get()
                break
            else:
                time.sleep(0.05)
            if event.is_set():
                break

    return
            
            
if __name__ == "__main__":
    
    
    datastream=queue.LifoQueue()
    event=threading.Event()
    event2=threading.Event()
    
    meas_thread=threading.Thread(target=meas)    
    meas_thread.start()
    
    time.sleep(1)
    control_thread=threading.Thread(target=control,daemon=False)
    control_thread.start()
    
    anim = animation.FuncAnimation(fig, animate, init_func=initialize_plot,
                                   frames=gen, interval=100,blit=False,cache_frame_data=False)
    plt.show()

    meas_thread.join()
    print("measurement thread finished")
    control_thread.join()
    print("control thread finished")