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

# CP, VCO, and Divider parameters
kpd = 5e-3
kvco = 30e6
n = 4500

# frequency analysis data
#f1 = 5.8e9 * 2 * np.pi # starting frequency
#f2 = 5.913e9 * 2 * np.pi # final frequency

# Fourth Order Loop filter parameters
c1 = 5.6e-9
c2 = 100e-9
c3 = 330e-12
c4 = 104e-12
r2 = 1000
r3 = 6800
r4 = 33000

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
s0 = k /n

# constants for accurate analysis
#q0 = (k * (f2-f1)) / n
#q1 = (k * (f2-f1) * t2) / n

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

def graph_noise(): # graph just phase noise
    plt.figure()
    plt.semilogx(x, tot)    # Bode magnitude plot
    plt.semilogx(x, losc, 'r')
    plt.grid()
    plt.ylim([-200,-50])
    plt.xlim([1e3,1e6])
    plt.gca().xaxis.grid(True, which='minor')
    plt.title('Phase Noise')
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'Phase Noise Power (dbc/Hz)')

#def graph_acc(): # graph just open loop gain response
#    # Open loop transfer response for PLL
#    graph_mag(x, mag_a, 'Accurate Closed Loop Response') 
#    graph_phase(x, phase_a, 'Accurate Closed Loop Response')
#    plt.show()

# numerators for transient response
num_filter = [t2, 1]
num_open = [t2_open, k]
num_closed = [s1, s0]
#num_accurate = [q1, q0]

# denominators for transient response
den_filter = [a3, a2, a1, a0, 0]
den_open = [a3, a2, a1, a0, 0, 0]
den_closed = [a3, a2, a1, a0, s1, s0]

# create transfer functions
f_filter = signal.TransferFunction(num_filter, den_filter)
f_open = signal.TransferFunction(num_open, den_open)
f_closed = signal.TransferFunction(num_closed, den_closed)
#f_accurate = signal.TransferFunction(num_accurate, den_closed)

# create plot data
x = np.arange (1000, 100e4, 10)                      # frequency range 0.1Hz to 100MHz
x_new = x * 2 * np.pi                               # change frequency to rads/s for bode plot function
w, mag, phase = signal.bode(f_filter, x_new)        # loop filter creation
w_o, mag_o, phase_o = signal.bode(f_open, x_new)    # open loop creation
w_c, mag_c, phase_c = signal.bode(f_closed, x_new)  # closed loop creation
#w_a, mag_a, phase_a = signal.bode(f_accurate, x_new)# more accurate response creation



# Phase Noise Simulation
losc = (-100.9) - (20 * np.log10( x / 10e3))
tot = mag_c + losc









