import numpy as np
import time
from multiprocessing import Pool
from scipy.optimize import minimize

###STAGE 1###
import sys
from config import CODE_DIRECTORY
sys.path.append(CODE_DIRECTORY)

# User settings for CUSP
import settings
from set_settings import *
from cusp_demo_utils import *
import cusp_stage1
import stage1_opt_data

print('#### STAGE 1 OF CUSP NOW RUNNING ####\n')

# Lists to store energies
check_energies = []       # Energies of FCI/exact wavefunctions
stage1_energies = []      # Energies of VQE wavefunctions

# Run thru bond lengths (or the training set)
for bond_length in bond_lengths:
    
    # Run VQE calculation for each training point/state
    opt_stage1_params = stage1_opt_data.run_state_preparation_optimization(bond_length)
    print('Optimizing for bond length {0} ... '
          'Optimal parameter setting is: {1}'.format(bond_length, opt_stage1_params))

    # Compute and store energies to check results
    exact_energy = settings.fetch_ground_energy(bond_length)
    check_energies.append(exact_energy)
    opt_energy = cusp_stage1.compute_stage1_cost_function(opt_stage1_params,
                                                          bond_length,
                                                          n_repetitions=num_trials,
                                                          exact=True,
                                                          noisy=include_gate_noise)
    stage1_energies.append(opt_energy)
    
    # Display stage 1 results
    print('Exact ground state energy               : {}'.format(exact_energy))
    print('VQE optimized energy                    : {}'.format(opt_energy))
    print('Energy difference (absolute value)      : {}\n'.format(
            np.abs(opt_energy - exact_energy)))
    
    # Save these optimized VQE parameters into numpy arrays
    np.save('data/stage1_param_{}'.format(bond_length), opt_stage1_params)