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
import control

# CP, VCO, and Divider parameters
kpd = 20e-6
kvco = 110e6
n = 63000

# frequency analysis data
f1 = 5.8e9 # starting frequency
f2 = 6.03e9 # final frequency

# Fourth Order Loop filter parameters
c1 = 1.52e-9
c2 = 37.7e-9
c3 = 893e-12
c4 = 252.3e-12
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
s1 = t2 * k
s0 = k

# constants for time analysis
q1 = (k * (f2-f1))/n
q0 = (k * (f2-f1) * t2)/n

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

def graph_time(): # function for graphing phase response
    plt.figure()
    plt.plot(time_t, time_f)    # Bode Phase plot
    plt.title(r'Time Response')
    plt.xlabel(r'time')
    plt.ylabel(r'Frequency (Hz)')


def main(): # function to create all 3 response graphs
    # transfer response of loop filter
    graph_mag(x, mag, 'Loop Filter Response') 
    graph_phase(x, phase, 'Loop Filter Response')
    # Open loop transfer response for PLL
    graph_mag(x, mag_o, 'Open Loop PLL Response') 
    graph_phase(x, phase_o, 'Open Loop PLL Response')
    # Closed loop transfer response for PLL
    graph_mag(x, mag_c, 'Closed Loop PLL Response') 
    graph_phase(x, phase_c, 'Closed Loop PLL Response')
    plt.show()

def graph_cl(): # graph just closed loop gain response
    # Closed loop transfer response for PLL
    graph_mag(x, mag_c, 'Closed Loop PLL Response') 
    graph_phase(x, phase_c, 'Closed Loop PLL Response')
    plt.show()

def graph_ol(): # graph just open loop gain response
    # Open loop transfer response for PLL
    graph_mag(x, mag_o, 'Open Loop PLL Response') 
    graph_phase(x, phase_o, 'Open Loop PLL Response')
    plt.show()

def graph_lf(): # graph just open loop gain response
    # Open loop transfer response for PLL
    graph_mag(x, mag, 'Loop Filter Response') 
    graph_phase(x, phase, 'Loop Filter Response')
    plt.show()

# numerators for transient response
num_filter = [t2, 1]
num_open = [t2_open, k]
num_closed = [s1, s0]
num_time = [q1, q0]

# denominators for transient response
den_filter = [a3, a2, a1, a0, 0]
den_open = [a3, a2, a1, a0, 0, 0]
den_closed = [a3, a2, a1, a0, s1, s0]

# create transfer functions
f_filter = signal.TransferFunction(num_filter, den_filter)
f_open = signal.TransferFunction(num_open, den_open)
f_closed = signal.TransferFunction(num_closed, den_closed)
f_time = control.tf(num_time, den_closed)

# create plot data
x = np.arange (0.1, 101e6, 10)                      # frequency range 0.1Hz to 100MHz
x_new = x * 2 * np.pi                               # change frequency to rads/s for bode plot function
w, mag, phase = signal.bode(f_filter, x_new)        # loop filter creation
w_o, mag_o, phase_o = signal.bode(f_open, x_new)    # open loop creation
w_c, mag_c, phase_c = signal.bode(f_closed, x_new)  # closed loop creation
time_t, time_f = control.step_response(f_time)      # time response creation





























