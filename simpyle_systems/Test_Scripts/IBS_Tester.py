"""
Author: socHACKi
This is to test the modem module as implementd for development
"""
# %%
import synthesizer
# %%
# File related information/settings
file_path = r'C:\Users\socHACKi\Desktop\temp'
# %%
# Analysis related informaiton/settings
reference_frequencies = [100, 200, 300]
cumulative_LO_mutiplication = 2
lo_tuning_range = 125
lo_frequency_dict = {'Rx': [5800 ,
                             5800 + 200,
                             5800 + 350,
                             5800 + 500,
                             5800 + 650,
                             5800 + 800,
                             5800 + 1000],
                      'Tx': [9933,
                             9933 + 200,
                             9933 + 350,
                             9933 + 500,
                             9933 + 650,
                             9933 + 800,
                             9933 + 1000]}
adjustment_dict = {'Rx': 5800,
                   'Tx': 5800}
cumulative_additive_noise_limit = 0.3
# %%
analysis = synthesizer.IntegerBoundarySpurs(
                                            file_path,
                                            reference_frequencies,
                                            cumulative_LO_mutiplication,
                                            lo_tuning_range,
                                            lo_frequency_dict,
                                            adjustment_dict,
                                            cumulative_additive_noise_limit)
# %%
analysis.spurious_plots_vs_synthesizer_output('display')
# %%
analysis.spurious_plots_vs_synthesizer_output('write', Type='png')
# %%
analysis.additive_noise_plots_vs_tuning_step_size('display')
# %%
analysis.additive_noise_plots_vs_tuning_step_size('write', Type='png')
# %%
analysis.additive_noise_plots_vs_synthesizer_output('display')
# %%
analysis.additive_noise_plots_vs_synthesizer_output('write', Type='png')
# %%
analysis.worst_case_additive_noise_plots_vs_tuning_step_size('display')
# %%
analysis.worst_case_additive_noise_plots_vs_tuning_step_size('write', Type='png')
# %%
analysis.write_results_to_excel()
