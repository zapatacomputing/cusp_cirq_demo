import numpy as np

### SETTINGS ###
num_trials = 1000
include_gate_noise = True
noise_level = 0.003

bond_lengths = [1.0,1.5,2.0,2.5]

user_settings = np.array([True, include_gate_noise, noise_level,
                          num_trials, bond_lengths], dtype=object)
np.save('data/user_settings', user_settings)
