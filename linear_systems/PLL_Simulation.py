# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:06:46 2017

@author: tmurphy

Model of PLL displaying open loop, closed loop, and loop filter response
outputting graphs of magnitude and phase for each.

Equations taken from PLL Performance, Simulation, and Design; 5th Edition; Dean Banerjee

loop filter should be correct.
open loop and closed loop appear accurate.
Time response not working right?
Is N correct?if not adjust kpd after fixing N?
Add phase noise plot if possible?
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math

# CP, VCO, and Divider parameters
kpd = 2.5e-3         # icp also
kvco = 110e6
n = 59.139

# Fourth Order Loop filter parameters
c1 = 1.52e-9
c2 = 37.7e-9
c3 = 893e-12
c4 = 253.3e-12
r2 = 145
r3 = 113
r4 = 738

# Constants used in transfer function of loop filter and open loop response
a0 = c1 + c2 + c3 + c4
a1 = c2 * r2 *(c1 + c3 + c4) + r3 * (c1 + c2) * (c3 + c4) + c4 * r4 *(c1 + c2 + c3)
a2 = c1 * c2 * r2 * r3 * (c3 +c4) + c4 * r4 * (c2 * c3 * r3 + c1 * c3 * r3 + c1 * c2 * r2 + c2 * c3 * r2)
a3 = c1 * c2 * c3 * c4 * r2 * r3 * r4
k = kpd * kvco
t2 = r2 * c2
t2_open = t2 * k

# adjusted constants for closed loop response
s1 = (t2 * k) / n
s0 = k / n


def graph_mag(w, mag, title): # function for graphing magnitude response
    plt.figure()
    plt.semilogx(w, mag)    # Bode magnitude plot
    plt.grid()
    plt.gca().xaxis.grid(True, which='minor')
    plt.title(title)
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'Magnitude (db)')

def graph_phase(w, phase, title): # function for graphing phase response
    plt.figure()
    plt.semilogx(w, phase)    # Bode Phase plot
    plt.grid()
    plt.gca().xaxis.grid(True, which='minor')
    plt.title(title)
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'Phase (deg)')

def graph_cl(): # graph just closed loop gain response
    # Closed loop transfer response for PLL
    graph_mag(freq, mag_c, 'Closed Loop PLL Response') 
    graph_phase(freq, phase_c, 'Closed Loop PLL Response')
    plt.show()

def graph_ol(): # graph just open loop gain response
    # Open loop transfer response for PLL
    graph_mag(freq, mag_o, 'Open Loop PLL Response') 
    graph_phase(freq, phase_o, 'Open Loop PLL Response')
    plt.show()

def graph_lf(): # graph just open loop gain response
    # Open loop transfer response for PLL
    graph_mag(freq, mag, 'Loop Filter Response') 
    graph_phase(freq, phase, 'Loop Filter Response')
    plt.show()

def graph_noise(): # graph just phase noise
    plt.figure()
    plt.semilogx(freq, vco_noise, 'b--')
    plt.semilogx(freq, ref_noise, 'g--')
    plt.semilogx(freq, s_ref, 'm')
    plt.semilogx(freq, vco_noise, 'r')
#    plt.semilogx(freq, r2_noise, 'r--')
#    plt.semilogx(freq, r3_noise, 'g--')
#    plt.semilogx(freq, r4_noise, 'b--')
#    plt.semilogx(freq, r_tot, 'm--')
    plt.grid()
    plt.ylim([-160,-60])
    plt.gca().xaxis.grid(True, which='minor')
    plt.title('Phase Noise')
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'Phase Noise Power (dbc/Hz)')
    plt.show()

#def r_noise(vx, tr): # equation to convert resistor noise from V to dBc/Hz
#    rout = (vx * kvco) / (freq * math.sqrt(2)) * abs(tr / [1 + mag_o/n])
#    return rout

# numerators for transient response
num_filter = [t2, 1]
num_open = [t2_open, k]
num_closed = [s1, s0]

# denominators for transient response
den_filter = [a3, a2, a1, a0, 0]
den_open = [a3, a2, a1, a0, 0, 0]
den_closed = [a3, a2, a1, a0, s1, s0]

# create transfer functions
f_filter = signal.TransferFunction(num_filter, den_filter)
f_open = signal.TransferFunction(num_open, den_open)
f_closed = signal.TransferFunction(num_closed, den_closed)

# create plot data
freq = np.arange (1, 100e6, 10)                      # frequency range 0.1Hz to 100MHz
x_new = freq * 2 * np.pi                               # change frequency to rads/s for bode plot function
w, mag, phase = signal.bode(f_filter, x_new)        # loop filter creation
w_o, mag_o, phase_o = signal.bode(f_open, x_new)    # open loop creation
w_c, mag_c, phase_c = signal.bode(f_closed, x_new)  # closed loop creation


# Phase Noise Simulation
ref_noise = np.piecewise(freq, [freq < 1, (10 > freq) & (freq >= 1), (100 > freq) & (freq >= 10), (1e3 > freq) & (freq >= 100), (1e4 > freq) & (freq >= 1e3), freq >= 1e4], [lambda x: -32 * np.log10(1241 * x), lambda x: -21 * np.log10(5.179e4 * x), lambda x: -17 * np.log10(1.145e6 * x), lambda x: -3 * np.log10(4.642e43 * x), lambda x: -5 * np.log10(1e25 * x), -145])
vco_noise = np.piecewise(freq, [freq < 1e4, (1e5 > freq) & (freq >= 1e4), (1e6 > freq) & (freq >= 1e5), (1e7 > freq) & (freq >= 1e6), (2e7 > freq) & (freq >= 1e7), (1e8 > freq) & (freq >= 2e7), freq >= 1e8], [lambda x: -30 * np.log10(0.1166 * x), lambda x: -23 * np.log10(x), lambda x: -21 * np.log10(2.994 * x), lambda x: -20 * np.log10(6.310 * x), lambda x: -19.93 * np.log10(6.711 * x), lambda x: 42.73 * 10**(-3.862e-8 * x) - 169.2, -196])
s_ref = 10*np.log10(mag_c)
s_vco = vco_noise * 1/np.asarray(mag_o+1)

# loop filter noise
v2 = math.sqrt(4 * 300 * 1.380658e-23 * r2)
v3 = math.sqrt(4 * 300 * 1.380658e-23 * r3)
v4 = math.sqrt(4 * 300 * 1.380658e-23 * r4)
z1 = signal.TransferFunction([c2*r2, 1], [c1*c2*r2, c1+c2, 0])
z2 = signal.TransferFunction([c3*c4*r3*r4, (c3*r3)+(c4*r4)+(r3*c4), 1],[c3*c4*r4, c3+c4, 0])
tmid = signal.TransferFunction(1, [c3*c4*r3*r4, (c3*r3)+(c4*r4)+(r3*c4), 1])
vn1, carry1, zn1= signal.bode(z1, x_new)
vn2, carry2, zn2= signal.bode(z2, x_new)
vn3, carry3, zn3= signal.bode(tmid, x_new)
tr2 = (carry3 * x_new * c2 * carry2) / (1 + x_new * [c2*r2 + c1*carry2 + c2*carry2] + x_new**2*c1*c2*r2*carry2)
tr3 = (carry3*carry2)/(carry1+carry2)
tr4 = (1 + x_new*c3*(r3+carry1)) / (1 + x_new*[(c3+c4)*(r3+carry1)+c4*r4] + x_new**2 * c3 * c4 * r4 * (r3+carry1))
r2_noise = np.transpose(r_noise(v2, tr2))
r3_noise = np.transpose(r_noise(v3, tr3))
r4_noise = np.transpose(r_noise(v4, tr4))
r_tot = 10*np.log10(10**(r2_noise/10) + 10**(r3_noise/10) + 10**(r4_noise/10))








