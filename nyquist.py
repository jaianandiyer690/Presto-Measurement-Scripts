import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Sampling settings
fs = 1000               # sampling frequency (Hz)
Ts = 1 / fs

# Time axis for continuous signal
t_cont = np.linspace(0, 5e-3, 5000)    # dense grid (analog)
t_samp = np.arange(0, 5e-3, Ts)        # sample points

# Prepare figure
fig, ax = plt.subplots(figsize=(8, 4))
line_cont, = ax.plot([], [], lw=2, label="Analog Signal")
line_samp, = ax.plot([], [], 'o', label="Sampled Points")
text_label = ax.text(0.02, 0.9, "", transform=ax.transAxes, fontsize=12)

ax.set_xlim(0, 5e-3)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()

# Frequency sweep values (0 → 2*Nyquist)
f_vals = np.linspace(0, 2*(fs/2), 200)

def animate(i):
    f = f_vals[i]
    # True analog wave
    x_cont = np.sin(2*np.pi*f*t_cont)
    x_samp = np.sin(2*np.pi*f*t_samp)

    # Update plots
    line_cont.set_data(t_cont, x_cont)
    line_samp.set_data(t_samp, x_samp)

    # Aliased frequency:
    # folding around Nyquist
    f_alias = abs((f + fs/2) % fs - fs/2)

    text_label.set_text(f"Input frequency: {f:6.1f} Hz\nAliased: {f_alias:6.1f} Hz")

    return line_cont, line_samp, text_label

ani = FuncAnimation(fig, animate, frames=len(f_vals), interval=50, blit=True)

plt.show()
