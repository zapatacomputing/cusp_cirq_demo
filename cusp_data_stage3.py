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
opt_qae_params = np.load('data/stage2_param.npy')

for bond_length in bond_lengths:
    stage1_param = np.load('data/stage1_param_{}.npy'.format(bond_length))
    stage1_param_list.append(stage1_param)
    
search_parameters_stage3 = ['aht','ht','zz']
fixed_aht = 0
fixed_ht = 0
fixed_zz = 0

user_parameters_stage3 = np.array([search_parameters_stage3, fixed_aht,
                                   fixed_ht, fixed_zz], dtype=object)
np.save('data/user_parameters_stage3', user_parameters_stage3)

import cusp_stage3
import stage3_opt_data

print('Parameters used from Stage 2: {}\n'.format(opt_qae_params))

print('#### STAGE 3 OF CUSP NOW RUNNING ####\n')

stage3_energies = []
cusp_params = {}

for i, bond_length in enumerate(bond_lengths):
    
    # Initialize parameters
    half_turn_min = 0
    half_turn_max = 2
    init_params = np.random.uniform(low=half_turn_min,
                                    high=half_turn_max,
                                    size=stage3_opt_data.num_param)

    result_list = []
    # Set up cost function
    def stage3_cost_function(lst, bond_length, N=num_trials):
        energy = stage3_opt_data.stage3(lst, bond_length, N)
        result_list.append(energy)
        print(energy)
        return energy
    
    # Optimization using Nelder-Mead
    stage3_fcn = lambda x: stage3_cost_function(x, bond_length=bond_length,
                                             N=num_trials)
    res = minimize(stage3_fcn, init_params, args=(),
                   method='Nelder-Mead', tol=None, 
                   options={'disp': False, 'maxiter': 100, 'xatol': 0.001,
                            'return_all': False, 'fatol': 0.001})
    opt_cusp_param = res.x
    opt_cusp_param = fix_list(opt_cusp_param, stage3_opt_data.all_param,stage3_opt_data.var_param,
                              stage3_opt_data.fixed_vals)
    cusp_params[bond_length] = opt_cusp_param
    cusp_energy = cusp_stage3.run_sim_repetitions_stage3(*opt_cusp_param,
                                                         bond_length=bond_length,
                                                         n_repetitions=num_trials,
                                                         exact=True,
                                                         noisy=include_gate_noise)
    stage3_energies.append(cusp_energy)
    np.savetxt('stage3_data_'+str(bond_length)+'.csv',result_list, delimiter=',')
    print('Bond length                             : {}'.format(bond_length))
    print('CUSP optimized energy                   : {}'.format(cusp_energy))
   # print('Stage 1 energy                          : {}'.format(stage1_energies[i]))
   # print('Exact energy                            : {}'.format(check_energies[i]))
   # print('Energy difference (Stage 1 vs. exact)   : {}'.format(
   #         np.abs(stage1_energies[i] - check_energies[i])))
   # print('Energy difference (CUSP    vs. exact)   : {}\n'.format(
   #         np.abs(cusp_energy - check_energies[i])))
