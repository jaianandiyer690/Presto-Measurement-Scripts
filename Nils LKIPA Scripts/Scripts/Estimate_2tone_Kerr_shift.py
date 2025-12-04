# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 14:13:31 2025

@author: Probestation
"""
import numpy as np

v_pump = 0.1
P_in = 1e-3*10**( (np.log10(v_pump**2) - 20.3 - 20 - 53)/10 )
print(P_in)
detune = 2*np.pi*3e6
omega_0 = 4.300e9
omega_p = omega_0 - detune
kappa_e = 2*np.pi*0.556*1.270*1e7
kappa_tot = 2*np.pi*1.357e7
h = 6.62e-34

Kerr_expected = -10

n = P_in/(h * omega_p) *  kappa_e / (kappa_tot**2/4 + detune**2)

print(f"intracavity photon number: {n}")
print(f"expected frequency shift: {n*Kerr_expected}")