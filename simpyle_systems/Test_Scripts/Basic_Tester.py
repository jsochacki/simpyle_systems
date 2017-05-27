"""
Author: socHACKi
This is to test the modem module as implementd for development
"""
# %%
import numpy as np
import os
#from socHACKi.socHACKiUtilityPackage import AttrDict

# %%
import matplotlib.pyplot as plt
# %%
# File related information/settings
file_path = r'C:\Users\socHACKi\Desktop\temp'
post_fix = '.csv'
# %%
# Analysis related informaiton/settings
# This is to adjust for the ADI software cutting the LO output frequency
# in half at some data point aside of what you have asked it to do
# All frequency values below the corresponding value below will be doubled.
# Their corresponding values will not as they are correct in their current
# state.
adjustment_dict = {'Rx': 5800,
                   'Tx': 5800}
reference_frequencies = [100, 200, 300]
cumulative_LO_mutiplication = 2
lo_frequency_dict = {'Rx': [5800 * cumulative_LO_mutiplication,
                             (5800 + 200) * cumulative_LO_mutiplication,
                             (5800 + 350) * cumulative_LO_mutiplication,
                             (5800 + 500) * cumulative_LO_mutiplication,
                             (5800 + 650) * cumulative_LO_mutiplication,
                             (5800 + 800) * cumulative_LO_mutiplication,
                             (5800 + 1000) * cumulative_LO_mutiplication],
                      'Tx': [9933 * cumulative_LO_mutiplication,
                             (9933 + 200) * cumulative_LO_mutiplication,
                             (9933 + 350) * cumulative_LO_mutiplication,
                             (9933 + 500) * cumulative_LO_mutiplication,
                             (9933 + 650) * cumulative_LO_mutiplication,
                             (9933 + 800) * cumulative_LO_mutiplication,
                             (9933 + 1000) * cumulative_LO_mutiplication]}
lo_tuning_range = 125
lo_tuning_step_sizes = np.arange(1,lo_tuning_range/cumulative_LO_mutiplication,1)
result_type_list = ['f_vals',
                    'dominant_IBS_spur_per_frequency',
                    'cumulative_addative_noise_per_frequency',
                    'effective_tuning_step_size',
                    'maximum_cumulative_addative_noise_per_step']
# %%
set_dict = {'Rx': reference_frequencies,
            'Tx': reference_frequencies}
results_dictionary = {}
# %%
available_files = os.listdir(file_path)
# %%
for Key in set_dict:
    reference_frequencies = set_dict[Key]
    results_dictionary.update({Key:{}})
    for ReferenceFrequency in reference_frequencies:
        for File in available_files:
            if (Key in File) \
               and (str(ReferenceFrequency) in File) \
               and (post_fix in File):
                results_dictionary[Key].update({ReferenceFrequency:{}})

                data_array = np.loadtxt(os.path.join(file_path,File),
                                        dtype=np.float32, delimiter=',',
                                        usecols=(1, 23, 25, 27, 28, 29),
                                        skiprows=8,
                                        unpack=True)

                analysis_synthesizer_frequency_values = data_array[0,:]

                IBS_1_values = np.array(data_array[1,:])
                IBS_2_values = np.array(data_array[2,:])
                IBS_3_values = np.array(data_array[3,:])
                IBS_4_values = np.array(data_array[4,:])
                IBS_5_values = np.array(data_array[5,:])

                analysis_synthesizer_frequency_values = np.array(
                    [Frequency
                         if (Frequency > adjustment_dict[Key])
                         else Frequency*2
                         for Frequency in analysis_synthesizer_frequency_values])

                synthesizer_frequency_values = \
                    analysis_synthesizer_frequency_values * \
                    cumulative_LO_mutiplication

                IBS_1_values = \
                    IBS_1_values + 20*np.log10(cumulative_LO_mutiplication)
                IBS_2_values = \
                    IBS_2_values + 20*np.log10(cumulative_LO_mutiplication)
                IBS_3_values = \
                    IBS_3_values + 20*np.log10(cumulative_LO_mutiplication)
                IBS_4_values = \
                    IBS_4_values + 20*np.log10(cumulative_LO_mutiplication)
                IBS_5_values = \
                    IBS_5_values + 20*np.log10(cumulative_LO_mutiplication)

                spurs_at_frequencies = np.array(
                        [[IBS_1_values[Index],
                          IBS_2_values[Index],
                          IBS_3_values[Index],
                          IBS_4_values[Index],
                          IBS_5_values[Index]]
                            for Index, Value in enumerate(IBS_1_values)]
                                               )
                dominant_ibs_spur = np.array(
                        [np.max(spurs_at_frequencies[Index])
                            for Index, Value in enumerate(spurs_at_frequencies)]
                                            )
                cumulative_addative_noise = \
                    np.sqrt(2 * np.sum(np.power(10,
                                                spurs_at_frequencies/10),
                                       axis=1)
                            ) / ((2 * np.pi) / 360)
                results_dictionary[Key][ReferenceFrequency].update(
                        {result_type_list[0]: synthesizer_frequency_values,
                         result_type_list[1]: dominant_ibs_spur,
                         result_type_list[2]: cumulative_addative_noise
                         })
# %%
effective_tuning_step_size = lo_tuning_step_sizes * cumulative_LO_mutiplication
for Key in set_dict:
    for ReferenceFrequency in reference_frequencies:
        for LoCenterFrequency in lo_frequency_dict[Key]:
            results_dictionary[
                    Key][
                    ReferenceFrequency].update({LoCenterFrequency:{}})
            max_addative_noise_array = []
            if LoCenterFrequency == np.min(lo_frequency_dict[Key]):
                for delta_frequency in effective_tuning_step_size:
                    max_addative_noise = 0
                    current_addative_noise = 0
                    for current_frequency in np.arange(LoCenterFrequency,
                                                       LoCenterFrequency + lo_tuning_range,
                                                       delta_frequency):

                        current_addative_noise = np.interp(current_frequency,
                                                           results_dictionary[
                                                               Key][
                                                               ReferenceFrequency][
                                                               result_type_list[0]],
                                                           results_dictionary[
                                                               Key][
                                                               ReferenceFrequency][
                                                               result_type_list[2]])

                        if (current_addative_noise > max_addative_noise):
                            max_addative_noise = current_addative_noise
                    max_addative_noise_array.append([delta_frequency, max_addative_noise])
                max_addative_noise_array = np.array(max_addative_noise_array)
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[3]: max_addative_noise_array[:,0]})
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[4]: max_addative_noise_array[:,1]})

# TODO I believe the error is in here
            elif LoCenterFrequency == np.max(lo_frequency_dict[Key]):
                for delta_frequency in effective_tuning_step_size:
                    max_addative_noise = 0
                    current_addative_noise = 0
                    for current_frequency in np.arange(LoCenterFrequency - lo_tuning_range,
                                                       LoCenterFrequency,
                                                       delta_frequency):

                        current_addative_noise = np.interp(current_frequency,
                                                           results_dictionary[
                                                               Key][
                                                               ReferenceFrequency][
                                                               result_type_list[0]],
                                                           results_dictionary[
                                                               Key][
                                                               ReferenceFrequency][
                                                               result_type_list[2]])

                        if (current_addative_noise > max_addative_noise):
                            max_addative_noise = current_addative_noise
                    max_addative_noise_array.append([delta_frequency, max_addative_noise])
                max_addative_noise_array = np.array(max_addative_noise_array)
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[3]: max_addative_noise_array[:,0]})
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[4]: max_addative_noise_array[:,1]})

            else:
                for delta_frequency in effective_tuning_step_size:
                    max_addative_noise = 0
                    current_addative_noise = 0
                    for current_frequency in np.arange(LoCenterFrequency - lo_tuning_range,
                                                       LoCenterFrequency + lo_tuning_range,
                                                       delta_frequency):

                        current_addative_noise = np.interp(current_frequency,
                                                           results_dictionary[
                                                               Key][
                                                               ReferenceFrequency][
                                                               result_type_list[0]],
                                                           results_dictionary[
                                                               Key][
                                                               ReferenceFrequency][
                                                               result_type_list[2]])

                        if (current_addative_noise > max_addative_noise):
                            max_addative_noise = current_addative_noise
                    max_addative_noise_array.append([delta_frequency, max_addative_noise])
                max_addative_noise_array = np.array(max_addative_noise_array)
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[3]: max_addative_noise_array[:,0]})
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[4]: max_addative_noise_array[:,1]})

# %%
for Key in set_dict:
    for ReferenceFrequency in reference_frequencies:
        plt.figure('{0} Synthesizer {1} results for '
                   '{2}MHz Reference Frequency'.format(
                Key,
                result_type_list[1],
                ReferenceFrequency))
        plt.plot(results_dictionary[Key][ReferenceFrequency][result_type_list[0]],
                 results_dictionary[Key][ReferenceFrequency][result_type_list[1]])
        plt.plot(results_dictionary[Key][ReferenceFrequency][result_type_list[0]],
                 results_dictionary[Key][ReferenceFrequency][result_type_list[2]])
# %%
for Key in set_dict:
    for ReferenceFrequency in reference_frequencies:
        plt.figure('{0} Synthesizer {1} results for '
                   '{2}MHz Reference Frequency'.format(
                Key,
                result_type_list[1],
                ReferenceFrequency))
        for LoCenterFrequency in lo_frequency_dict[Key]:
            plt.plot(results_dictionary[Key][ReferenceFrequency][LoCenterFrequency][result_type_list[3]],
                     results_dictionary[Key][ReferenceFrequency][LoCenterFrequency][result_type_list[4]])
# %%
plt.plot(results_dictionary[Key][100][54665]['effective_tuning_step_size'],results_dictionary[Key][100][54665]['maximum_cumulative_addative_noise_per_step'])
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