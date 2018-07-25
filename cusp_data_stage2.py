import numpy as np
import time
from multiprocessing import Pool
from scipy.optimize import minimize

###STAGE 2###
import sys
from config import CODE_DIRECTORY
sys.path.append(CODE_DIRECTORY)

# User settings for CUSP
import settings
from set_settings import *
from cusp_demo_utils import *

# Load the optimized VQE parameters from stage 1 of CUSP
stage1_param_list = []

for bond_length in bond_lengths:
    stage1_param = np.load('data/stage1_param_{}.npy'.format(bond_length))
    stage1_param_list.append(stage1_param)
    
search_parameters_stage2 = ['w1', 'w2', 'z', 'cz']
fixed_w1 = .25
fixed_w2 = .5
fixed_z = 1
fixed_cz = 1

user_parameters_stage2 = np.array([search_parameters_stage2, fixed_w1, fixed_w2,
                          fixed_z, fixed_cz], dtype=object)
np.save('data/user_parameters_stage2', user_parameters_stage2)

import cusp_stage2
import stage2_opt_data

print('Stage 2 using the following bond lengths for training: {}\n'.format(bond_lengths))

# QAE settings
threshold = 0.2
n_qae_trials = 10

print('#### STAGE 2 OF CUSP NOW RUNNING ####\n')
opt_qae_params = stage2_opt_data.run_qae_optimization(training_states=stage1_param_list,
                                                 n_repetitions=num_trials,
                                                 exact=True,
                                                 noisy=include_gate_noise)

# Repeat optimization of QAE circuit while error value is above threshold
iter_count = 0
while stage2_opt_data.compute_avg_fid_proxy(params=opt_qae_params,
                                       training_states=stage1_param_list,
                                       n_repetitions=num_trials,
                                       exact=True,
                                       noisy=include_gate_noise) > threshold:
    if iter_count >= n_qae_trials:
        print('Surpassed the QAE iteration limit. Exiting loop.')
        break
    
    print('Trial {}: Quantum autoencoder learning had low fidelity. '
          'Trying again.'.format(iter_count))
    
    opt_qae_params = stage2_opt_data.run_qae_optimization(training_states=stage1_param_list,
                                                     n_repetitions=num_trials,
                                                     exact=True,
                                                     noisy=include_gate_noise)
    iter_count += 1

# Compute error of optimized QAE circuit
err = stage2_opt_data.compute_avg_fid_proxy(opt_qae_params, training_states=stage1_param_list,
                                       n_repetitions=num_trials, exact=True,
                                       noisy=include_gate_noise)
print('Quantum autoencoder learning succeeded with error : {}'.format(err))

opt_qae_params = fix_list(opt_qae_params, stage2_opt_data.all_param,stage2_opt_data.var_param,
                          stage2_opt_data.fixed_vals)
# Save QAE results
np.save('data/stage2_param', opt_qae_params)
print('')