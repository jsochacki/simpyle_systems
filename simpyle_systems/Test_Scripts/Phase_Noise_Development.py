"""
Author: socHACKi
This class is to be used for linear modeling of PLL systems.  There are a few
architectures available for the pll's that are modeled by this class.
"""

import numpy as np

from socHACKi.socHACKiUtilityPackage import AttrDict
from simpyle_systems.synthesizer import PhaseNoise

# %%
import matplotlib.pyplot as plt
# %%
# Synthesizer Values
Kv = 121 # In MHZ / Volt
Kp = 0.000397887 # In Amps / Rad
phase_detector_FOM = -239
phase_detector_frequency = 10000000
N = 3280 # Feedback divider setting
M = 1 # Reference Divider Setting
# %%
# Phase Noise Information
phase_noise_offset_frequencies_hz = [0.1, 1, 10, 100, 1000,
                                     10000, 100000, 1000000, 10000000]
#loop_filter_output_voltage_noise =
loop_filter_output_phase_noise = \
    PhaseNoise([[0.1,-144], [1,-165], [10,-167],
                [100,-168], [1000,-168], [10000,-169]],
               0,
               50)
vco_phase_noise = \
    PhaseNoise([[100,-30], [1000,-60], [10000,-91],
                [100000,-114], [1000000,-134], [10000000,-153]],
               5.8482e9,
               50)

reference_phase_noise = \
    PhaseNoise([[0.1,-87], [1,-119], [10,-140],
                [100,-157], [1000,-160], [10000,-165]],
               10e6,
               50)
# %%
# Loop Filter Component Values
C1 = 37.1e-9
C2 = 681e-9
C3 = 24.2e-9
C4 = 7.8e-9
R2 = 31.7
R3 = 19.9
R4 = 110
# %%
# Passive Loop FIlter Model
# Generate Passive Loop Filter Coefficients From Component Values

# +------+                                                           +--------+
# |      |                          R3                 R4            |  Kvco  |
# | Kpd  +-------+----------+----+/\/\/\+-----+-----+/\/\/\+----+----+   ---  |
# |      |       |          |                 |                 |    |    s   |
# +------+       |          |                 |                 |    +--------+
#                +          +                 +                 +
#               ___        ___               ___               ___
#            C1 ___     C2 ___            C3 ___            C4 ___
#                +          +                 +                 +
#                |          |                 |                 |
#                +          +                 +                 +
#               _|_         \                _|_               _|_
#               \ /      R2 /                \ /               \ /
#                -          \                 -                 -
#                           /
#                          _|_
#                          \ /
#                           -

A0 = C1 + C2 + C3 + C4

A1 = (C2 * R2 * (C1 + C3 + C4)) + \
     (R3 * (C1 + C2) * (C3 + C4)) + \
     (C4 * R4 * (C1 + C2 + C3))

A2 = ((C1 * C2 * R2 * R3 * (C3 + C4)) + \
      (C4 * R4 * ((C2 * C3 * R3) + \
                  (C1 * C3 * R3) + \
                  (C1 * C2 * R2) + \
                  (C2 * C3 * R2)
                  )
      )
     )

A3 = C1 * C2 * C3 * C4 * R2 * R3 * R4

T2 = R2 * C2
 # %%
 # Synthesizer Functions
def loop_filter_transfer_impedance(frequency, T2, A3, A2, A1, A0):
    s = 1j * 2 * np.pi * frequency
    return ((1 + (s * T2)) /
            (s * ((A3 * np.power(s, 3)) +
                  (A2 * np.power(s, 2)) +
                  (A1 * s) +
                  (A0)
                 )
            )
           )

def open_loop_transfer_function(frequency, Z, Kp, Kv):
    s = 1j * 2 * np.pi * frequency
    KV = Kv * 2 * np.pi * 1e6
    KP = Kp
    return ((KV * KP * Z) / s)

def loop_filter_transfer_function(G, N, Z, Kp):
    KP = Kp
    return ((G / (KP * Z)) / (1 - (G / N)))

def charge_pump_transfer_function(G, N, Kp):
    KP = 50#Kp
    return ((G / KP) / (1 - (G / N)))

def vco_transfer_function(G, N):
    return (1 / (1 - (G / N)))

def reference_transfer_function(G, N):
    return (G / (1 - (G / N)))
# %%
def generate_phase_detector_phase_noise(FOffset, FReference, FCarrier, FOM, FDetector):
    output = [FOffset]
    output.append([FOM +
                   10*np.log10(FDetector) +
                   20*np.log10(FCarrier / FReference)]
                   * len(FOffset))
    return PhaseNoise.pair(output[0],output[1])
# %%
start_frequency = 0.1
stop_frequency = 1e6
frequency = np.array(PhaseNoise.logspace(start_frequency, int(np.log10(stop_frequency/start_frequency))))
# frequency = np.array(range(start_frequency,stop_frequency))
# %%
Z = loop_filter_transfer_impedance(frequency,T2, A3, A2, A1, A0)
OLTF = open_loop_transfer_function(frequency, Z, Kv, Kp)
LFTF = loop_filter_transfer_function(OLTF, N, Z, Kp)
CPTF = charge_pump_transfer_function(OLTF, N, Kp)
VCOTF = vco_transfer_function(OLTF, N)
REFTF = reference_transfer_function(OLTF, N)
# %%
fig = plt.figure(1)
ax = fig.add_subplot(111)
# ax.set_yscale('log')
ax.set_xscale('log')
ax.plot(frequency, 20*np.log10(np.abs(Z)), color='blue', lw=2)

# %%
ax.plot(frequency, 20*np.log10(np.abs(OLTF)), color='red', lw=2)
ax.plot(frequency, 20*np.log10(np.abs(VCOTF)), color='gold', lw=2)
ax.plot(frequency, 20*np.log10(np.abs(LFTF)), color='indigo', lw=2)
ax.plot(frequency, 20*np.log10(np.abs(CPTF)), color='tan', lw=2)
ax.plot(frequency, 20*np.log10(np.abs(REFTF)), color='black', lw=2)
fig.show()
# %%
f_offset = np.array(phase_noise_offset_frequencies_hz)
# f_offset = np.array(logspace(0.1,8))
Zvvf = PhaseNoise.pair(f_offset,
            loop_filter_transfer_impedance(f_offset,
                                           T2, A3, A2, A1, A0))
OLTFvvf = PhaseNoise.pair(f_offset,
               open_loop_transfer_function(f_offset,
                                           np.array(PhaseNoise.split(Zvvf)[1]),
                                           Kv, Kp))
LFTFvvf = PhaseNoise.pair(f_offset,
               loop_filter_transfer_function(np.array(PhaseNoise.split(OLTFvvf)[1]),
                                             N,
                                             np.array(PhaseNoise.split(Zvvf)[1]),
                                             Kp))
CPTFvvf = PhaseNoise.pair(f_offset,
               charge_pump_transfer_function(np.array(PhaseNoise.split(OLTFvvf)[1]),
                                             N,
                                             Kp))
VCOTFvvf = PhaseNoise.pair(f_offset,
                vco_transfer_function(np.array(PhaseNoise.split(OLTFvvf)[1]),
                                      N))
REFTFvvf = PhaseNoise.pair(f_offset,
                reference_transfer_function(np.array(PhaseNoise.split(OLTFvvf)[1]),
                                            N))


Zvvf_dB = []
OLTFvvf_dB = []
LFTFvvf_dB = []
CPTFvvf_dB = []
VCOTFvvf_dB = []
REFTFvvf_dB = []
Zvvf_dB.append(PhaseNoise.split(Zvvf)[0])
OLTFvvf_dB.append(PhaseNoise.split(OLTFvvf)[0])
LFTFvvf_dB.append(PhaseNoise.split(LFTFvvf)[0])
CPTFvvf_dB.append(PhaseNoise.split(CPTFvvf)[0])
VCOTFvvf_dB.append(PhaseNoise.split(VCOTFvvf)[0])
REFTFvvf_dB.append(PhaseNoise.split(REFTFvvf)[0])
Zvvf_dB.append(20*np.log10(np.abs(PhaseNoise.split(Zvvf)[1])))
OLTFvvf_dB.append(20*np.log10(np.abs(PhaseNoise.split(OLTFvvf)[1])))
LFTFvvf_dB.append(20*np.log10(np.abs(PhaseNoise.split(LFTFvvf)[1])))
CPTFvvf_dB.append(20*np.log10(np.abs(PhaseNoise.split(CPTFvvf)[1])))
VCOTFvvf_dB.append(20*np.log10(np.abs(PhaseNoise.split(VCOTFvvf)[1])))
REFTFvvf_dB.append(20*np.log10(np.abs(PhaseNoise.split(REFTFvvf)[1])))
# %%
fig = plt.figure(2)
ax = fig.add_subplot(111)
# ax.set_yscale('log')
ax.set_xscale('log')
ax.plot(Zvvf_dB[0], Zvvf_dB[1], color='blue', lw=2)

# %%
ax.plot(OLTFvvf_dB[0], OLTFvvf_dB[1],
        color='red', lw=2)
ax.plot(LFTFvvf_dB[0], LFTFvvf_dB[1],
        color='gold', lw=2)
ax.plot(CPTFvvf_dB[0], CPTFvvf_dB[1],
        color='indigo', lw=2)
ax.plot(VCOTFvvf_dB[0], VCOTFvvf_dB[1],
        color='tan', lw=2)
ax.plot(REFTFvvf_dB[0], REFTFvvf_dB[1],
        color='black', lw=2)
fig.show()
# %%
# Do phase noise formatting and filling
phase_detector_phase_noise_at_10GHz_dBm = \
    generate_phase_detector_phase_noise(phase_noise_offset_frequencies_hz,
                                        10000000,
                                        32.8e9,
                                        phase_detector_FOM,
                                        phase_detector_frequency)
reference_phase_noise.change_frequency(32.8e9)
vco_phase_noise.change_frequency(32.8e9)
loop_filter_output_phase_noise.phase_noise_fill(phase_noise_offset_frequencies_hz, [])
vco_phase_noise.phase_noise_fill(phase_noise_offset_frequencies_hz, [])
reference_phase_noise.phase_noise_fill(phase_noise_offset_frequencies_hz, [])

# %%
# Calculate results
LFPN = PhaseNoise.split(PhaseNoise.combine(PhaseNoise.pair(LFTFvvf_dB[0], LFTFvvf_dB[1]),
                     loop_filter_output_phase_noise.phase_noise))
PDPN = PhaseNoise.split(PhaseNoise.combine(PhaseNoise.pair(CPTFvvf_dB[0], CPTFvvf_dB[1]),
                     phase_detector_phase_noise_at_10GHz_dBm))
VCOPN = PhaseNoise.split(PhaseNoise.combine(PhaseNoise.pair(VCOTFvvf_dB[0], VCOTFvvf_dB[1]),
                     vco_phase_noise.phase_noise))
REFPN = PhaseNoise.split(PhaseNoise.combine(PhaseNoise.pair(REFTFvvf_dB[0], REFTFvvf_dB[1]),
                     reference_phase_noise.phase_noise))
# %%
# Plot results
fig = plt.figure('Phase Noise Results')
ax = fig.add_subplot(111)
# ax.set_yscale('log')
ax.set_xscale('log')
ax.plot(loop_filter_output_phase_noise.phase_noise_split[0],
        loop_filter_output_phase_noise.phase_noise_split[1],
        color='red', linestyle='--', lw=1)
ax.plot(PhaseNoise.split(phase_detector_phase_noise_at_10GHz_dBm)[0],
        PhaseNoise.split(phase_detector_phase_noise_at_10GHz_dBm)[1],
        color='green', linestyle='--', lw=1)
ax.plot(vco_phase_noise.phase_noise_split[0],
        vco_phase_noise.phase_noise_split[1],
        color='blue', linestyle='--', lw=1)
ax.plot(reference_phase_noise.phase_noise_split[0],
        reference_phase_noise.phase_noise_split[1],
        color='black', linestyle='--', lw=2)
ax.plot(VCOPN[0], VCOPN[1],
        color='blue', lw=1)
ax.plot(PDPN[0], PDPN[1],
        color='green', lw=1)
ax.plot(LFPN[0], LFPN[1],
        color='red', lw=1)
ax.plot(REFPN[0], REFPN[1],
        color='black', lw=1)
fig.show()