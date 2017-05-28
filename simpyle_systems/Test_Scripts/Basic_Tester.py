"""
Author: socHACKi
This is to test the modem module as implementd for development
"""
# %%
import numpy as np
import os
import pandas as pd
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
cumulative_addative_noise_limit = 0.3
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
result_type_list = ['f_vals_MHz',
                    'dominant_IBS_spur_per_frequency',
                    'cumulative_addative_noise_per_frequency',
                    'effective_tuning_step_size_MHz',
                    'maximum_cumulative_addative_noise_per_step',
                    'range_tuned',
                    'center_frequency_hit',
                    'worst_case_maximum_cumulative_addative_noise_per_step']
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
                         if (Frequency >= adjustment_dict[Key])
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
            current_tuning_range_array = []
            center_frequency_hit_array = []
            if LoCenterFrequency == np.min(lo_frequency_dict[Key]):
                for delta_frequency in effective_tuning_step_size:
                    max_addative_noise = 0
                    current_addative_noise = 0
                    current_tuning_range = delta_frequency * (
                                    lo_tuning_range // delta_frequency)
                    center_frequency_hit = False
                    # Need to add the + delta_frequency to the top end of the
                    # list to get actually specified range due to lack of
                    # inclusion on range end in python, grrrrr!! :(
                    for current_frequency in np.arange(
                            LoCenterFrequency,
                            LoCenterFrequency + current_tuning_range +
                                delta_frequency,
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

                        if ~center_frequency_hit:
                            if current_frequency == LoCenterFrequency:
                                center_frequency_hit = True

                    max_addative_noise_array.append([delta_frequency, max_addative_noise])
                    current_tuning_range_array.extend([current_tuning_range])
                    center_frequency_hit_array.extend([center_frequency_hit])

                max_addative_noise_array = np.array(max_addative_noise_array)
                current_tuning_range_array = np.array(current_tuning_range_array)
                center_frequency_hit_array = np.array(center_frequency_hit_array)

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
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[5]: current_tuning_range_array})
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[6]: center_frequency_hit_array})

            elif LoCenterFrequency == np.max(lo_frequency_dict[Key]):
                for delta_frequency in effective_tuning_step_size:
                    max_addative_noise = 0
                    current_addative_noise = 0
                    current_tuning_range = delta_frequency * (
                                    lo_tuning_range // delta_frequency)
                    center_frequency_hit = False
                    # Need to add the + delta_frequency to the top end of the
                    # list to get actually specified range due to lack of
                    # inclusion on range end in python, grrrrr!! :(
                    for current_frequency in np.arange(
                            LoCenterFrequency - current_tuning_range,
                            LoCenterFrequency + delta_frequency,
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

                        if ~center_frequency_hit:
                            if current_frequency == LoCenterFrequency:
                                center_frequency_hit = True

                    max_addative_noise_array.append([delta_frequency, max_addative_noise])
                    current_tuning_range_array.extend([current_tuning_range])
                    center_frequency_hit_array.extend([center_frequency_hit])

                max_addative_noise_array = np.array(max_addative_noise_array)
                current_tuning_range_array = np.array(current_tuning_range_array)
                center_frequency_hit_array = np.array(center_frequency_hit_array)

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
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[5]: current_tuning_range_array})
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[6]: center_frequency_hit_array})

            else:
                for delta_frequency in effective_tuning_step_size:
                    max_addative_noise = 0
                    current_addative_noise = 0
                    current_tuning_range = delta_frequency * (
                                    lo_tuning_range // delta_frequency)
                    center_frequency_hit = False
                    # Need to add the + delta_frequency to the top end of the
                    # list to get actually specified range due to lack of
                    # inclusion on range end in python, grrrrr!! :(
                    for current_frequency in np.arange(
                            LoCenterFrequency - current_tuning_range,
                            LoCenterFrequency + current_tuning_range +
                                delta_frequency,
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

                        if ~center_frequency_hit:
                            if current_frequency == LoCenterFrequency:
                                center_frequency_hit = True

                    max_addative_noise_array.append([delta_frequency, max_addative_noise])
                    current_tuning_range_array.extend([current_tuning_range])
                    center_frequency_hit_array.extend([center_frequency_hit])

                max_addative_noise_array = np.array(max_addative_noise_array)
                current_tuning_range_array = np.array(current_tuning_range_array)
                center_frequency_hit_array = np.array(center_frequency_hit_array)

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
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[5]: current_tuning_range_array})
                results_dictionary[
                        Key][
                        ReferenceFrequency][
                        LoCenterFrequency].update(
                            {result_type_list[6]: center_frequency_hit_array})
# %%
for Key in set_dict:
    for ReferenceFrequency in reference_frequencies:
        worst_case_maximum_array = []
        for Index in range(len(effective_tuning_step_size)):
            worst_case_maximum = 0
            current_worst_case_maximum = 0
            for LoCenterFrequency in lo_frequency_dict[Key]:
                current_worst_case_maximum = results_dictionary[
                                                Key][
                                                ReferenceFrequency][
                                                LoCenterFrequency][
                                                result_type_list[4]][Index]
                if current_worst_case_maximum > worst_case_maximum:
                    worst_case_maximum = current_worst_case_maximum

            worst_case_maximum_array.extend([worst_case_maximum])

        worst_case_maximum_array = np.array(worst_case_maximum_array)
        results_dictionary[
                    Key][
                    ReferenceFrequency].update(
                        {result_type_list[3]: effective_tuning_step_size})
        results_dictionary[
                    Key][
                    ReferenceFrequency].update(
                        {result_type_list[7]: worst_case_maximum_array})
# %%
plt.close("all")
for Key in set_dict:
    for ReferenceFrequency in reference_frequencies:
        fig = plt.figure('{0} Synthesizer {1} results for '
                         '{2}MHz Reference Frequency'.format(
                                 Key,
                                 result_type_list[1],
                                 ReferenceFrequency),
                        figsize=(18,10))
        plt.title('{0} Synthesizer IBS results for '
                   '{2}MHz Reference Frequency'.format(
                Key,
                result_type_list[1],
                ReferenceFrequency))

        plt.plot(results_dictionary[Key][ReferenceFrequency][result_type_list[0]],
                 results_dictionary[Key][ReferenceFrequency][result_type_list[1]],
                 label='Dominant IBS Spur Value')
        plt.legend()

        plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0.)

        plt.grid(color='k', linestyle='--', linewidth=0.5)
        plt.grid(which='major', alpha=0.9)

        resolution = 20
        size = results_dictionary[Key][ReferenceFrequency][result_type_list[0]].size
        min_x = results_dictionary[Key][ReferenceFrequency][result_type_list[0]][0]
        max_x = results_dictionary[Key][ReferenceFrequency][result_type_list[0]][-1]
        plt.xticks(np.arange(min_x, max_x, ((max_x - min_x) // resolution)))

        ax = fig.add_subplot(1, 1, 1)

        ax.set_xticks(results_dictionary[
                        Key][
                        ReferenceFrequency][
                        result_type_list[0]][0::(size // (resolution * 2))],
                      minor=True)

        plt.grid(which='minor', color='k', linestyle='--', linewidth=0.5)
        plt.grid(which='minor', alpha=0.3)

        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Dominant IBS Spur Value (dBc)')

        plt.savefig(os.path.join(file_path,
                               results_directory_name,
                               'Dominant_IBS_Spur_Value_'
                               '{0}_Synthesizer_{1}MHz_Ref_vs_tuneF'.format(
                                       Key,
                                       ReferenceFrequency)),
                    bbox_inches='tight')

# %%
plt.close("all")
for Key in set_dict:
    for ReferenceFrequency in reference_frequencies:
        fig = plt.figure('{0} Synthesizer {1} results for '
                         '{2}MHz Reference Frequency'.format(
                                 Key,
                                 result_type_list[2],
                                 ReferenceFrequency),
                         figsize=(18,10))
        plt.title('{0} Synthesizer Additive results for '
                   '{2}MHz Reference Frequency'.format(
                Key,
                result_type_list[1],
                ReferenceFrequency))

        plt.plot(results_dictionary[Key][ReferenceFrequency][result_type_list[0]],
                 results_dictionary[Key][ReferenceFrequency][result_type_list[2]],
                 label='Cumulative Addative Noise Due To IBSs')
        plt.legend()

        plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0.)

        plt.grid(color='k', linestyle='--', linewidth=0.5)
        plt.grid(which='major', alpha=0.9)

        resolution = 20
        size = results_dictionary[Key][ReferenceFrequency][result_type_list[0]].size
        min_x = results_dictionary[Key][ReferenceFrequency][result_type_list[0]][0]
        max_x = results_dictionary[Key][ReferenceFrequency][result_type_list[0]][-1]
        plt.xticks(np.arange(min_x, max_x, ((max_x - min_x) // resolution)))

        ax = fig.add_subplot(1, 1, 1)

        ax.set_xticks(results_dictionary[
                        Key][
                        ReferenceFrequency][
                        result_type_list[0]][0::(size // (resolution * 2))],
                      minor=True)

        plt.grid(which='minor', color='k', linestyle='--', linewidth=0.5)
        plt.grid(which='minor', alpha=0.3)

        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Cumulative Additive Noise (degrees RMS)')

        plt.savefig(os.path.join(file_path,
                               results_directory_name,
                               'Cumulative_Additive_IBS_Noise_'
                               '{0}_Synthesizer_{1}MHz_Ref_vs_tuneF'.format(
                                       Key,
                                       ReferenceFrequency)),
                    bbox_inches='tight')
# %%
plt.close("all")
for Key in set_dict:
    for ReferenceFrequency in reference_frequencies:
        fig = plt.figure('{0} Synthesizer {1} results for '
                         '{2}MHz Reference Frequency'.format(
                        Key,
                        result_type_list[1],
                        ReferenceFrequency),
                        figsize=(18,10))
        plt.title('Cumulative Additive Noise Due To Integer Boundary Spurs\n'
                  '{0} Synthesizer {1}MHz Reference Frequency Results'.format(
                        Key,
                        ReferenceFrequency))

        for LoCenterFrequency in lo_frequency_dict[Key]:
            plt.plot(results_dictionary[Key][ReferenceFrequency][LoCenterFrequency][result_type_list[3]],
                     results_dictionary[Key][ReferenceFrequency][LoCenterFrequency][result_type_list[4]],
                     '-o',
                     label=('{0}MHz Synthesizer '
                            'Center Frequency'.format(LoCenterFrequency)))
            plt.legend()
        plt.plot(results_dictionary[Key][ReferenceFrequency][LoCenterFrequency][result_type_list[3]],
                 cumulative_addative_noise_limit *
                 np.ones(results_dictionary[Key][ReferenceFrequency][LoCenterFrequency][result_type_list[3]].size),
                 label='Cumulative Additive Noise Limit')
        plt.legend()

        plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0.)

        plt.grid(color='k', linestyle='--', linewidth=0.5)
        plt.grid(which='major', alpha=0.9)

        resolution = 20
        min_x = results_dictionary[Key][ReferenceFrequency][result_type_list[3]][0]
        max_x = results_dictionary[Key][ReferenceFrequency][result_type_list[3]][-1]
        plt.xticks(np.arange(min_x, max_x, ((max_x - min_x) // resolution)))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(results_dictionary[Key][ReferenceFrequency][result_type_list[3]],
                      minor=True)

        plt.grid(which='minor', color='k', linestyle='--', linewidth=0.5)
        plt.grid(which='minor', alpha=0.3)

        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Cumulative Additive Noise (degrees RMS)')

        plt.savefig(os.path.join(file_path,
                               results_directory_name,
                               'Cumulative_Additive_IBS_Noise_'
                               '{0}_Synthesizer_{1}MHz_Ref_vs_tune_step'.format(
                                       Key,
                                       ReferenceFrequency)),
                    bbox_inches='tight')
# %%
plt.close("all")
for Key in set_dict:
    for ReferenceFrequency in reference_frequencies:
        fig = plt.figure('{0} Synthesizer Worst Case results for '
                         '{1}MHz Reference Frequency'.format(
                        Key,
                        ReferenceFrequency),
                        figsize=(18,10))
        plt.title('Cumulative Additive Noise Due To Integer Boundary Spurs\n'
                  '{0} Synthesizer {1}MHz Reference Frequency Results'.format(
                        Key,
                        ReferenceFrequency))

        plt.plot(results_dictionary[Key][ReferenceFrequency][result_type_list[3]],
                 results_dictionary[Key][ReferenceFrequency][result_type_list[7]],
                 '-o',
                 label=('Limiting Synthesizer Case Only'))
        plt.legend()

        plt.plot(results_dictionary[Key][ReferenceFrequency][result_type_list[3]],
                 cumulative_addative_noise_limit *
                 np.ones(results_dictionary[Key][ReferenceFrequency][result_type_list[3]].size),
                 label='Cumulative Additive Noise Limit')
        plt.legend()

        plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0.)

        plt.grid(color='k', linestyle='--', linewidth=0.5)
        plt.grid(which='major', alpha=0.9)

        resolution = 20
        min_x = results_dictionary[Key][ReferenceFrequency][result_type_list[3]][0]
        max_x = results_dictionary[Key][ReferenceFrequency][result_type_list[3]][-1]
        plt.xticks(np.arange(min_x, max_x, ((max_x - min_x) // resolution)))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(results_dictionary[Key][ReferenceFrequency][result_type_list[3]],
                      minor=True)

        plt.grid(which='minor', color='k', linestyle='--', linewidth=0.5)
        plt.grid(which='minor', alpha=0.3)

        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Cumulative Additive Noise (degrees RMS)')

        plt.savefig(os.path.join(file_path,
                               results_directory_name,
                               'Cumulative_Worst_Case_Additive_IBS_Noise_'
                               '{0}_Synthesizer_{1}MHz_Ref_vs_tune_step'.format(
                                       Key,
                                       ReferenceFrequency)),
                    bbox_inches='tight')
# %%
results_directory_name = None
dir_count = 0
while(results_directory_name == None):
    try:
        os.listdir(os.path.join(file_path,'Results_{0}'.format(dir_count)))
    except FileNotFoundError as e:
        os.mkdir(os.path.join(file_path,'Results_{0}'.format(dir_count)))
        results_directory_name = 'Results_{0}'.format(dir_count)
    except PermissionError as e:
        print('L1 PermissionError: ',e)
        dir_count = dir_count + 1
    else:
        try:
            os.rmdir(os.path.join(file_path,'Results_{0}'.format(dir_count)))
        except OSError as e:
            print('L2 OSError: ',e)
            dir_count = dir_count + 1
        except PermissionError as e:
            print('L2 PermissionError: ',e)
            dir_count = dir_count + 1
        else:
           try:
               os.mkdir(os.path.join(file_path,'Results_{0}'.format(dir_count)))
           except PermissionError as e:
               print('L3 PermissionError: ',e)
               dir_count = dir_count + 1
           else:
               results_directory_name = 'Results_{0}'.format(dir_count)

# %%
dataframe_dict = {}
for Key in set_dict:
    for ReferenceFrequency in reference_frequencies:
        series_dict = {}
        series_dict.update({result_type_list[3]:
            pd.Series(results_dictionary[
                        Key][
                        ReferenceFrequency][
                        result_type_list[3]],
                      name=result_type_list[3])})
        series_dict.update({result_type_list[7]:
            pd.Series(results_dictionary[
                        Key][
                        ReferenceFrequency][
                        result_type_list[7]],
                      name=result_type_list[7])})
        for LoCenterFrequency in lo_frequency_dict[Key]:
            series_dict.update({'Additive Noise '
                                'Results_for_{0}MHz'.format(LoCenterFrequency):
                pd.Series(results_dictionary[
                            Key][
                            ReferenceFrequency][
                            LoCenterFrequency][
                            result_type_list[4]],
                          name='Additive Noise '
                                'Results_for_{0}MHz'.format(LoCenterFrequency))})
        dataframe_dict.update({'AdditiveNoise{0}With'
                                   '{1}MHzRef'.format(Key,
                                                       ReferenceFrequency):
                               pd.DataFrame.from_dict(series_dict).set_index(
                                   [result_type_list[3]])
                               })
        order = dataframe_dict['AdditiveNoise{0}With'
                               '{1}MHzRef'.format(Key,
                                                  ReferenceFrequency)
                              ].columns.tolist()
        order = order[-1:] + order[:-1]
        dataframe_dict['AdditiveNoise{0}With'
                       '{1}MHzRef'.format(Key,
                                          ReferenceFrequency)
                      ] = dataframe_dict['AdditiveNoise{0}With'
                                         '{1}MHzRef'.format(Key,
                                                            ReferenceFrequency)
                                        ].ix[:, order]
# %%
dataframe_dict_2 = {}
for Key in set_dict:
    for ReferenceFrequency in reference_frequencies:
        series_dict = {}
        series_dict.update({result_type_list[0]:
            pd.Series(results_dictionary[
                        Key][
                        ReferenceFrequency][
                        result_type_list[0]],
                      name=result_type_list[0])})
        series_dict.update({result_type_list[1]:
            pd.Series(results_dictionary[
                        Key][
                        ReferenceFrequency][
                        result_type_list[1]],
                      name=result_type_list[1])})
        dataframe_dict_2.update({'Spurious{0}With'
                                   '{1}MHzRef'.format(Key,
                                                       ReferenceFrequency):
                               pd.DataFrame.from_dict(series_dict).set_index(
                                   [result_type_list[0]])
                               })
# %%
for Key in dataframe_dict:
    writer = pd.ExcelWriter(os.path.join(file_path,
                               results_directory_name,
                               '{0}.xlsx'.format(Key)),
                            engine='xlsxwriter')
    dataframe_dict[Key].to_excel(writer, sheet_name=Key)
    writer.save()
for Key in dataframe_dict_2:
    writer = pd.ExcelWriter(os.path.join(file_path,
                               results_directory_name,
                               '{0}.xlsx'.format(Key)),
                            engine='xlsxwriter')
    dataframe_dict_2[Key].to_excel(writer, sheet_name=Key)
    writer.save()
# %%