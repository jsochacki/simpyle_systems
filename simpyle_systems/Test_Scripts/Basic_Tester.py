"""
Author: socHACKi
This is to test the modem module as implementd for development
"""
# %%
import numpy as np
from socHACKi.socHACKiUtilityPackage import AttrDict

# %%
import matplotlib.pyplot as plt
# %%
constellation = AttrDict()

# Create the bpsk attributes
constellation._bpsk = AttrDict()

constellation._bpsk.ring_ratio = 1

constellation._bpsk.decimal_to_symbol_mapping = \
    np.array(list(((decimal_value, symbol_location)
                    for decimal_value, symbol_location in enumerate([0, 1])
                  ))
            )

constellation._bpsk.symbol_to_decimal_mapping = \
    np.fliplr(constellation._bpsk.decimal_to_symbol_mapping)

constellation._bpsk.decimal_alphabet = \
    constellation._bpsk.decimal_to_symbol_mapping[::, 0]

constellation._bpsk.complex_alphabet = \
    np.exp(1j * (np.pi * constellation._bpsk.decimal_alphabet))

constellation._bpsk.binary_alphabet = \
    np.array([bin(i) for i in constellation._bpsk.decimal_alphabet])

constellation._bpsk.n_symbols = constellation._bpsk.complex_alphabet.size

constellation._bpsk.bits_per_word = int(np.log2(constellation._bpsk.n_symbols))

constellation._bpsk.n_rings = 1

constellation._bpsk.MODCOD = 1


# Create the qpsk attributes
constellation._qpsk = AttrDict()

constellation._qpsk.ring_ratio = 1

constellation._qpsk.decimal_to_symbol_mapping = \
    np.array(list(((decimal_value, symbol_location)
                    for decimal_value, symbol_location in
                        enumerate([0, 3, 1, 2])
                  ))
            )

constellation._qpsk.symbol_to_decimal_mapping = \
    np.fliplr(constellation._qpsk.decimal_to_symbol_mapping)

constellation._qpsk.decimal_alphabet = \
    constellation._qpsk.decimal_to_symbol_mapping[::, 0]

constellation._qpsk.complex_alphabet = \
    np.exp(1j * (((np.pi / 2) * constellation._qpsk.decimal_alphabet) +
                 (np.pi / 4)))

constellation._qpsk.binary_alphabet = \
    np.array([bin(i) for i in constellation._qpsk.decimal_alphabet])

constellation._qpsk.n_symbols = constellation._qpsk.complex_alphabet.size

constellation._qpsk.bits_per_word = int(np.log2(constellation._qpsk.n_symbols))

constellation._qpsk.n_rings = 1

constellation._qpsk.MODCOD = 2


# Create the 8psk attributes
constellation._8psk = AttrDict()

constellation._8psk.ring_ratio = 1

constellation._8psk.decimal_to_symbol_mapping = \
    np.array(list(((decimal_value, symbol_location)
                    for decimal_value, symbol_location in
                        enumerate([0, 7, 3, 4, 1, 6, 2, 5])
                  ))
            )

constellation._8psk.symbol_to_decimal_mapping = \
    np.fliplr(constellation._8psk.decimal_to_symbol_mapping)

constellation._8psk.decimal_alphabet = \
    constellation._8psk.decimal_to_symbol_mapping[::, 0]

constellation._8psk.complex_alphabet = \
    np.exp(1j * (((np.pi / 4) * constellation._8psk.decimal_alphabet) +
                 (np.pi / 4)))

constellation._8psk.binary_alphabet = \
    np.array([bin(i) for i in constellation._8psk.decimal_alphabet])

constellation._8psk.n_symbols = constellation._8psk.complex_alphabet.size

constellation._8psk.bits_per_word = int(np.log2(constellation._8psk.n_symbols))

constellation._8psk.n_rings = 1

constellation._8psk.MODCOD = 3


# Create the 16apsk Ring Ratio 3.15 attributes
constellation._16apsk_rr3p15 = AttrDict()

constellation._16apsk_rr3p15.ring_ratio = 3.15

constellation._16apsk_rr3p15.decimal_to_symbol_mapping = \
    np.array(list(((decimal_value, symbol_location)
                    for decimal_value, symbol_location in
                        enumerate(list(range(0,16,1)))
                  ))
            )

constellation._16apsk_rr3p15.symbol_to_decimal_mapping = \
    np.fliplr(constellation._16apsk_rr3p15.decimal_to_symbol_mapping)

constellation._16apsk_rr3p15.decimal_alphabet = \
    constellation._16apsk_rr3p15.decimal_to_symbol_mapping[::, 0]

constellation._16apsk_rr3p15.complex_alphabet = np.hstack((
    constellation._16apsk_rr3p15.ring_ratio *
        np.exp(1j * (((np.pi / 6) *
                     np.array([0, 9, 3, 6, 11, 10, 4, 5, 1, 8, 2, 7])) +
                    (np.pi / 4))),
    np.exp(1j * (((np.pi / 2) * np.array([0, 3, 1, 2])) +
                 (np.pi / 4)))))

constellation._16apsk_rr3p15.binary_alphabet = \
    np.array([bin(i) for i in constellation._16apsk_rr3p15.decimal_alphabet])

constellation._16apsk_rr3p15.n_symbols = \
    constellation._16apsk_rr3p15.complex_alphabet.size

constellation._16apsk_rr3p15.bits_per_word = \
    int(np.log2(constellation._16apsk_rr3p15.n_symbols))

constellation._16apsk_rr3p15.n_rings = 2

constellation._16apsk_rr3p15.MODCOD = 4


# Create the 16apsk Ring Ratio 2.85 attributes
constellation._16apsk_rr2p85 = AttrDict()

constellation._16apsk_rr2p85.ring_ratio = 2.85

constellation._16apsk_rr2p85.decimal_to_symbol_mapping = \
    np.array(list(((decimal_value, symbol_location)
                    for decimal_value, symbol_location in
                        enumerate(list(range(0,16,1)))
                  ))
            )

constellation._16apsk_rr2p85.symbol_to_decimal_mapping = \
    np.fliplr(constellation._16apsk_rr2p85.decimal_to_symbol_mapping)

constellation._16apsk_rr2p85.decimal_alphabet = \
    constellation._16apsk_rr2p85.decimal_to_symbol_mapping[::, 0]

constellation._16apsk_rr2p85.complex_alphabet = np.hstack((
    constellation._16apsk_rr2p85.ring_ratio *
        np.exp(1j * (((np.pi / 6) *
                     np.array([0, 9, 3, 6, 11, 10, 4, 5, 1, 8, 2, 7])) +
                    (np.pi / 4))),
    np.exp(1j * (((np.pi / 2) * np.array([0, 3, 1, 2])) +
                 (np.pi / 4)))))

constellation._16apsk_rr2p85.binary_alphabet = \
    np.array([bin(i) for i in constellation._16apsk_rr2p85.decimal_alphabet])

constellation._16apsk_rr2p85.n_symbols = \
    constellation._16apsk_rr2p85.complex_alphabet.size

constellation._16apsk_rr2p85.bits_per_word = \
    int(np.log2(constellation._16apsk_rr2p85.n_symbols))

constellation._16apsk_rr2p85.n_rings = 2

constellation._16apsk_rr2p85.MODCOD = 5


# Create the 16apsk Ring Ratio 2.75 attributes
constellation._16apsk_rr2p75 = AttrDict()

constellation._16apsk_rr2p75.ring_ratio = 2.75

constellation._16apsk_rr2p75.decimal_to_symbol_mapping = \
    np.array(list(((decimal_value, symbol_location)
                    for decimal_value, symbol_location in
                        enumerate(list(range(0,16,1)))
                  ))
            )

constellation._16apsk_rr2p75.symbol_to_decimal_mapping = \
    np.fliplr(constellation._16apsk_rr2p75.decimal_to_symbol_mapping)

constellation._16apsk_rr2p75.decimal_alphabet = \
    constellation._16apsk_rr2p75.decimal_to_symbol_mapping[::, 0]

constellation._16apsk_rr2p75.complex_alphabet = np.hstack((
    constellation._16apsk_rr2p75.ring_ratio *
        np.exp(1j * (((np.pi / 6) *
                     np.array([0, 9, 3, 6, 11, 10, 4, 5, 1, 8, 2, 7])) +
                    (np.pi / 4))),
    np.exp(1j * (((np.pi / 2) * np.array([0, 3, 1, 2])) +
                 (np.pi / 4)))))

constellation._16apsk_rr2p75.binary_alphabet = \
    np.array([bin(i) for i in constellation._16apsk_rr2p75.decimal_alphabet])

constellation._16apsk_rr2p75.n_symbols = \
    constellation._16apsk_rr2p75.complex_alphabet.size

constellation._16apsk_rr2p75.bits_per_word = \
    int(np.log2(constellation._16apsk_rr2p75.n_symbols))

constellation._16apsk_rr2p75.n_rings = 2

constellation._16apsk_rr2p75.MODCOD = 6


# Create the 16apsk Ring Ratio 2.70 attributes
constellation._16apsk_rr2p70 = AttrDict()

constellation._16apsk_rr2p70.ring_ratio = 2.70

constellation._16apsk_rr2p70.decimal_to_symbol_mapping = \
    np.array(list(((decimal_value, symbol_location)
                    for decimal_value, symbol_location in
                        enumerate(list(range(0,16,1)))
                  ))
            )

constellation._16apsk_rr2p70.symbol_to_decimal_mapping = \
    np.fliplr(constellation._16apsk_rr2p70.decimal_to_symbol_mapping)

constellation._16apsk_rr2p70.decimal_alphabet = \
    constellation._16apsk_rr2p70.decimal_to_symbol_mapping[::, 0]

constellation._16apsk_rr2p70.complex_alphabet = np.hstack((
    constellation._16apsk_rr2p70.ring_ratio *
        np.exp(1j * (((np.pi / 6) *
                     np.array([0, 9, 3, 6, 11, 10, 4, 5, 1, 8, 2, 7])) +
                    (np.pi / 4))),
    np.exp(1j * (((np.pi / 2) * np.array([0, 3, 1, 2])) +
                 (np.pi / 4)))))

constellation._16apsk_rr2p70.binary_alphabet = \
    np.array([bin(i) for i in constellation._16apsk_rr2p70.decimal_alphabet])

constellation._16apsk_rr2p70.n_symbols = \
    constellation._16apsk_rr2p70.complex_alphabet.size

constellation._16apsk_rr2p70.bits_per_word = \
    int(np.log2(constellation._16apsk_rr2p70.n_symbols))

constellation._16apsk_rr2p70.n_rings = 2

constellation._16apsk_rr2p70.MODCOD = 7


# Create the 16apsk Ring Ratio 2.60 attributes
constellation._16apsk_rr2p60 = AttrDict()

constellation._16apsk_rr2p60.ring_ratio = 2.60

constellation._16apsk_rr2p60.decimal_to_symbol_mapping = \
    np.array(list(((decimal_value, symbol_location)
                    for decimal_value, symbol_location in
                        enumerate(list(range(0,16,1)))
                  ))
            )

constellation._16apsk_rr2p60.symbol_to_decimal_mapping = \
    np.fliplr(constellation._16apsk_rr2p60.decimal_to_symbol_mapping)

constellation._16apsk_rr2p60.decimal_alphabet = \
    constellation._16apsk_rr2p60.decimal_to_symbol_mapping[::, 0]

constellation._16apsk_rr2p60.complex_alphabet = np.hstack((
    constellation._16apsk_rr2p60.ring_ratio *
        np.exp(1j * (((np.pi / 6) *
                     np.array([0, 9, 3, 6, 11, 10, 4, 5, 1, 8, 2, 7])) +
                    (np.pi / 4))),
    np.exp(1j * (((np.pi / 2) * np.array([0, 3, 1, 2])) +
                 (np.pi / 4)))))

constellation._16apsk_rr2p60.binary_alphabet = \
    np.array([bin(i) for i in constellation._16apsk_rr2p60.decimal_alphabet])

constellation._16apsk_rr2p60.n_symbols = \
    constellation._16apsk_rr2p60.complex_alphabet.size

constellation._16apsk_rr2p60.bits_per_word = \
    int(np.log2(constellation._16apsk_rr2p60.n_symbols))

constellation._16apsk_rr2p60.n_rings = 2

constellation._16apsk_rr2p60.MODCOD = 8


# Create the 16apsk Ring Ratio 2.57 attributes
constellation._16apsk_rr2p87 = AttrDict()

constellation._16apsk_rr2p87.ring_ratio = 2.57

constellation._16apsk_rr2p87.decimal_to_symbol_mapping = \
    np.array(list(((decimal_value, symbol_location)
                    for decimal_value, symbol_location in
                        enumerate(list(range(0,16,1)))
                  ))
            )

constellation._16apsk_rr2p87.symbol_to_decimal_mapping = \
    np.fliplr(constellation._16apsk_rr2p87.decimal_to_symbol_mapping)

constellation._16apsk_rr2p87.decimal_alphabet = \
    constellation._16apsk_rr2p87.decimal_to_symbol_mapping[::, 0]

constellation._16apsk_rr2p87.complex_alphabet = np.hstack((
    constellation._16apsk_rr2p87.ring_ratio *
        np.exp(1j * (((np.pi / 6) *
                     np.array([0, 9, 3, 6, 11, 10, 4, 5, 1, 8, 2, 7])) +
                    (np.pi / 4))),
    np.exp(1j * (((np.pi / 2) * np.array([0, 3, 1, 2])) +
                 (np.pi / 4)))))

constellation._16apsk_rr2p87.binary_alphabet = \
    np.array([bin(i) for i in constellation._16apsk_rr2p87.decimal_alphabet])

constellation._16apsk_rr2p87.n_symbols = \
    constellation._16apsk_rr2p87.complex_alphabet.size

constellation._16apsk_rr2p87.bits_per_word = \
    int(np.log2(constellation._16apsk_rr2p87.n_symbols))

constellation._16apsk_rr2p87.n_rings = 2

constellation._16apsk_rr2p87.MODCOD = 9


# Create the 16apsk Ring Ratio X.XX attributes
constellation._16apsk_rrxpxx = AttrDict()

constellation._16apsk_rrxpxx.ring_ratio = 7

constellation._16apsk_rrxpxx.decimal_to_symbol_mapping = \
    np.array(list(((decimal_value, symbol_location)
                    for decimal_value, symbol_location in
                        enumerate(list(range(0,16,1)))
                  ))
            )

constellation._16apsk_rrxpxx.symbol_to_decimal_mapping = \
    np.fliplr(constellation._16apsk_rrxpxx.decimal_to_symbol_mapping)

constellation._16apsk_rrxpxx.decimal_alphabet = \
    constellation._16apsk_rrxpxx.decimal_to_symbol_mapping[::, 0]

constellation._16apsk_rrxpxx.complex_alphabet = np.hstack((
    constellation._16apsk_rrxpxx.ring_ratio *
        np.exp(1j * (((np.pi / 6) *
                     np.array([0, 9, 3, 6, 11, 10, 4, 5, 1, 8, 2, 7])) +
                    (np.pi / 4))),
    np.exp(1j * (((np.pi / 2) * np.array([0, 3, 1, 2])) +
                 (np.pi / 4)))))

constellation._16apsk_rrxpxx.binary_alphabet = \
    np.array([bin(i) for i in constellation._16apsk_rrxpxx.decimal_alphabet])

constellation._16apsk_rrxpxx.n_symbols = \
    constellation._16apsk_rrxpxx.complex_alphabet.size

constellation._16apsk_rrxpxx.bits_per_word = \
    int(np.log2(constellation._16apsk_rrxpxx.n_symbols))

constellation._16apsk_rrxpxx.n_rings = 2

constellation._16apsk_rrxpxx.MODCOD = 10

# %%
# Make a method of the class called show constellation eventually
fig, ax = plt.subplots()
x_values = m.constellation._16apsk_rr3p15.complex_alphabet.real
y_values = m.constellation._16apsk_rr3p15.complex_alphabet.imag
ax.scatter(x_values, y_values)
for point, value in enumerate(m.constellation._16apsk_rr3p15.decimal_alphabet):
    ax.annotate(m.constellation._16apsk_rr3p15.symbol_to_decimal_mapping[point,0], (x_values[point], y_values[point]+0.15))
    ax.annotate(m.constellation._16apsk_rr3p15.binary_alphabet[m.constellation._16apsk_rr3p15.symbol_to_decimal_mapping[point,0]], (x_values[point], y_values[point]+0.05))
# %%
# Make a method of the class called plot symbolstream eventually
plt.scatter(m.symbol_stream.real,m.symbol_stream.imag)
# %%
plt.close()
# %%
m = Modem('bpsk')
m.generate_pulse_shaping_filter('firrcos', 24, 0.25, 8)
h =m.firrcos
hh = [item for item in h.flatten()]
import matplotlib.pyplot as plt
plt.plot(hh)
# %%
%%matlab -o mat_h
symbols = 24
USAMPR = 8
Rolloff=0.25
Order = symbols*USAMPR
mat_h = firrcos(Order, 0.5, Rolloff, USAMPR, 'rolloff', 'sqrt')
mat_h = mat_h.*24*Rolloff
# %%
h = mat_h[0]
plt.plot(h)