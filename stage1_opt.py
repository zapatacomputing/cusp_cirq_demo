
"""Driver routine for Stage One of CUSP: Training Set Preparation."""

import settings
import cusp_stage1
import numpy as np
from scipy.optimize import minimize

from cusp_demo_utils import *


# Settings for running the QAE optimization
num_trials = settings.num_trials
bond_lengths = settings.bond_lengths
no_noise = settings.no_sampling_noise
gate_error = settings.gate_error
alpha = 'alpha'
var_param = [alpha]

# A wrapper to make sure that the selection of var_param above is properly resolved when input to the function
num_param = len(var_param)
fixed_vals = [0, 0, 0]
all_param = [alpha]


def run_state_preparation_optimization(bond_length):
    """Runs optimization for VQE or state preparation.

    Args:
    =====
    bond_length : float
        Bond length of system-of-interest

    Returns:
    ========
    optimized_params : numpy.ndarray
        Vector of optimized state preparation parameters
    """
    # Initialize parameters
    half_turn_min = 0
    half_turn_max = 2
    init_params = np.random.uniform(low=half_turn_min, high=half_turn_max, size=num_param)

    # Set up cost function
    def stage1(lst, N=num_trials):
        input_list = fix_list(lst, all_param_array=all_param, var_param_array=var_param, fixed_vals_array=fixed_vals)
        energy = cusp_stage1.compute_stage1_cost_function(*input_list, bond_length, n_repetitions=N,
                                                          exact=no_noise, noisy=gate_error)
        return energy

    # Minimize using Nelder-Mead
    res = minimize(stage1, init_params, args=(),
                   method='Nelder-Mead', tol=None, 
                   options={'disp': False, 'maxiter': 30,
                   'xatol': 0.001, 'return_all': False, 'fatol': 0.001})
    return res.x
