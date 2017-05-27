"""
Author: socHACKi
This module is a collection of matlab like functions that allow the user
to run matlab like code and do communications work in python.
"""

import numpy as np
# import scipy as sp

from socHACKi.socHACKiUtilityPackage import AttrDict

class Modem(object):
    """
    This is a modem class that containes all of the information and methods
    that are necessary to modulate and demodulate a signal

    Parameters
    ----------

        Modulation Type : String
                          'BPSK', 'QPSK', '8PSK', or '16APSK'

        Ring Ratio : float, optional
                     If not specified this is assumed to be 1, else
                     set this equal to the ring ratio of the constellation
                     you are looking for.

                     If this is a standard ring ratio Per ETSI302307
                     you will get a unique attribute for the constellation,
                     else you will get a generic one that will have xpxx
                     in the place of the ring ratio attribute value.

    Returns
    -------

        object : modem object
                 This object is the object that will do all the work and
                 present the user a source or act as a sink

    See Also
    --------

    Nothing currently

    Example
    -------

    >>> instance = Modem('BPSK')
    <simpyle_comms.modem.Modem at 0xab73a58>

    instance is in instance of a simple BPSK modem object now

    """
    def __init__(self, *args):
        self._available_modcods = ['BPSK', 'QPSK', '8PSK', '16APSK']
        # FIXME Obviously I need to add dynamic argparsing here but for now
        self._constellation = self.initialize_constellation(args[0], 3.15)

    @property
    def available_modcods(self):
        return self._available_modcods

    @property
    def constellation(self):
        return self._constellation

    @available_modcods.setter
    def available_modcods(self):
        pass

    def generate_pulse_shaping_filter(self,
                                      Type,
                                      FilterLengthInSymbols,
                                      RolloffRate,
                                      DigitalOverSamplingRate):
        """
        Generates a pulse shaping filter based on the Type specified
        and the provided parameters

        Parameters
        ----------

        Type : String
               Equal to the type of filter that you want to generate
               in the modem.  Available options are:
                   'firrcos', ....

        FilterLengthInSymbols : int
                                equal to the number of symbols that
                                the filter should act on

        RolloffRate : float
                      The Rate at which the filter rolls off.
                      Also viewed as the excess bandwidth of the filter
                      relative to the DigitalOverSamplingRate

        DigitalOverSamplingRate : int
                                  The samplign rate of the filter
                                  to be generated

        Returns
        -------

        NONE

        See Also
        --------

        Nothing currently

        Example
        -------

        >>> m = Modem('BPSK')
        >>> USAMPR = 2 # THis is the upsampling rate from critically sampled
        >>> m.generate_random_symbol_stream(1024)
        >>> m.upsample(USAMPR)
        >>> m.generate_pulse_shaping_filter('firrcos', 24, 0.25, USAMPR)
        >>> m.firrcos

        This yeilds the time domain values of the filters impulse response
        """
        if Type.lower() == 'firrcos':
            Order = FilterLengthInSymbols * DigitalOverSamplingRate
            if Order % 2:
                Order = Order + 1
                print(('The FilterLengthInSymbols and DigitalOverSamplingRate\n'
                       'that was provided made the filter Order odd so the\n'
                       'order was increased by 1'))
            SymbolRate = 1
            Ts = 1 / SymbolRate
            Fc = SymbolRate / 2
            time_step = 1 / DigitalOverSamplingRate
            firrcos = np.zeros(int(Order/2) + 1, dtype=np.complex128)
            firrcos[0] = (1 / Ts) * \
                        (1 - RolloffRate + ((4 * RolloffRate) / np.pi))
            for index in range(1, len(firrcos)):
                tx = (index * Ts) / DigitalOverSamplingRate
                if tx == (1 / (4 * RolloffRate)):
                    firrcos[index] = (RolloffRate / (Ts * np.sqrt(2))) * \
                        (
                          ((1 + (2 / np.pi)) *
                           np.sin(np.pi / (4 * RolloffRate))) +
                          ((1 - (2 / np.pi)) *
                           np.cos(np.pi / (4 * RolloffRate)))
                        )
                else:
                    firrcos[index] = (1/Ts) * \
                        (
                         ( np.sin(np.pi * (tx * (1 - RolloffRate))) +
                           (4 * RolloffRate * tx * \
                               np.cos(np.pi * (tx * (1 + RolloffRate))))
                         ) /
                         (np.pi * tx *
                             (1 - np.power(4 * RolloffRate * tx, 2))
                         )
                        )
            self.firrcos = np.hstack([firrcos[-1:0:-1],firrcos])
#        elif Type.lower() == 'firrcosm':
#            Order = FilterLengthInSymbols * DigitalOverSamplingRate
#            if Order % 2:
#                Order = Order + 1
#                print(('The FilterLengthInSymbols and DigitalOverSamplingRate\n'
#                       'that was provided made the filter Order odd so the\n'
#                       'order was increased by 1'))
#            SymbolRate = 1
#            Ts = 1 / SymbolRate
#            Fc = SymbolRate / 2
#            time_step = 1 / DigitalOverSamplingRate
#            firrcos = np.zeros(Order, dtype=np.complex128)

    def upsample(self, USAMPR):
        """
        Upsample symbol_stream attribute by USAMPR

        Parameters
        ----------

        USAMPR : int
                 Upsampling Factor

        NONE
        self.symbol_stream : 1D np.array
                             The modems symbol_stream attribute is upsampled
                             in place

        Returns
        -------

        NONE
        self.symbol_stream : 1D np.array
                             The upsampled version of the symbol_stream
                             attribute
        """
        result = np.zeros(len(self.symbol_stream) * USAMPR,
                          dtype=np.complex128)
        result[0::USAMPR] = self.symbol_stream
        self.symbol_stream = result

    def generate_random_symbol_stream(self, NumberOfSymbols):
        self.symbol_stream = np.random.choice(self.constellation[
                list(self.constellation.keys())[0]].complex_alphabet,
                                              NumberOfSymbols)

    def initialize_constellation(self, ModCod, RingRatio):
        constellation = AttrDict()

        if ModCod.upper() == 'BPSK':
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

        elif ModCod.upper() == 'QPSK':
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

        elif ModCod.upper() == '8PSK':
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

        elif ModCod.upper() == '16APSK':
            if RingRatio == 3.15:
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

            elif RingRatio == 2.85:
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

            elif RingRatio == 2.75:
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

            elif RingRatio == 2.70:
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

            elif RingRatio == 2.60:
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

            elif RingRatio == 2.57:
                # Create the 16apsk Ring Ratio 2.57 attributes
                constellation._16apsk_rr2p57 = AttrDict()

                constellation._16apsk_rr2p57.ring_ratio = 2.57

                constellation._16apsk_rr2p57.decimal_to_symbol_mapping = \
                    np.array(list(((decimal_value, symbol_location)
                                    for decimal_value, symbol_location in
                                        enumerate(list(range(0,16,1)))
                                  ))
                            )

                constellation._16apsk_rr2p57.symbol_to_decimal_mapping = \
                    np.fliplr(constellation._16apsk_rr2p57.decimal_to_symbol_mapping)

                constellation._16apsk_rr2p57.decimal_alphabet = \
                    constellation._16apsk_rr2p57.decimal_to_symbol_mapping[::, 0]

                constellation._16apsk_rr2p57.complex_alphabet = np.hstack((
                    constellation._16apsk_rr2p57.ring_ratio *
                        np.exp(1j * (((np.pi / 6) *
                                     np.array([0, 9, 3, 6, 11, 10, 4, 5, 1, 8, 2, 7])) +
                                    (np.pi / 4))),
                    np.exp(1j * (((np.pi / 2) * np.array([0, 3, 1, 2])) +
                                 (np.pi / 4)))))

                constellation._16apsk_rr2p57.binary_alphabet = \
                    np.array([bin(i) for i in constellation._16apsk_rr2p57.decimal_alphabet])

                constellation._16apsk_rr2p57.n_symbols = \
                    constellation._16apsk_rr2p57.complex_alphabet.size

                constellation._16apsk_rr2p57.bits_per_word = \
                    int(np.log2(constellation._16apsk_rr2p57.n_symbols))

                constellation._16apsk_rr2p57.n_rings = 2

                constellation._16apsk_rr2p57.MODCOD = 9

            else:
                # Create the 16apsk Ring Ratio X.XX attributes
                constellation._16apsk_rrxpxx = AttrDict()

                constellation._16apsk_rrxpxx.ring_ratio = RingRatio

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

        return constellation


class Functions(object):
    """
    Unimplemented currently
    """
    def __init__(self, *args):
        self.pi = np.pi

    @staticmethod
    def upsample(symbol_stream, USAMPR):
        """
        Upsample provided symbol stream by USAMPR

        Parameters
        ----------

        USAMPR : int
                 Upsampling Factor

        symbol_stream : 1D np.array
                        An external provided symbol_stream

        Returns
        -------

        symbol_stream : 1D np.array
                        The upsampled version of the provided
                        symbol_stream
        """
        result = np.zeros(len(symbol_stream) * USAMPR,
                          dtype=np.complex128)
        result[0::USAMPR] = symbol_stream
        return result

    @property
    def document_name(self):
        pass

    @staticmethod
    def circ_shift_2_to_1(x, y):
        pass

    @staticmethod
    def xcorr(x, y):
        if len(x) < len(y):
            x, y = y, x
        return np.correlate(x, y, mode='full')

    @staticmethod
    def linear_convolution(x, y):
        # Theirs is so much faster and I have checked that it provides the
        # exact same answer as cls.linear_convolution(x, y) so just use theirs
        # theirs is np.convolve(x, y)
        y = np.array(y)
        if len(x) < len(y):
            x, y = y, x
        N = len(x)
        M = len(y)
        x = np.array(sum([[0] * (M-1), list(x), [0] * (M-1)], []))
        result = np.zeros(shape=(M + N - 1, ))
        for Index in range(0, M + N - 1):
            result[Index] = sum(y * x[range(Index + M - 1, Index - 1, -1)])
        return result


class Constants(object):
    """
    Unimplemented currently
    """
    def __init__(self, *args):
        self.pi = np.pi

# Required imports if put in sepatate package
# from Sochacki.SochackiUtilityPackage import AttrDict
# from Sochacki.SochackiUtilityPackage import StringListCollector