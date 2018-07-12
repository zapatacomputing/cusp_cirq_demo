
"""Driver routine for Stage Three of CUSP: Generative Model Search."""

import cusp_stage3
from scipy.optimize import minimize

import settings
from cusp_demo_utils import *


# Settings
user_parameters_stage3 = np.load('data/user_parameters_stage3.npy')
num_trials = settings.num_trials
no_noise = settings.no_sampling_noise
gate_error = settings.gate_error
var_param = user_parameters_stage3[0]

# A wrapper to make sure that the selection of var_param above is properly resolved when input to the function
num_param = len(var_param)
fixed_vals = [user_parameters_stage3[1], user_parameters_stage3[2], user_parameters_stage3[3]]
all_param = ['aht','ht','zz']

def stage3(lst, bond_length, n_repetitions=num_trials):
    """Cost function for stage 3. Outputs the energy expectation for given training set point.
    
    Args:
    =====
    lst : list or numpy.ndarray
        Vector of parameters for decoding circuit
    bond_length : float
        Bond length
    n_repetitions : int
        Number of circuit trials

    Returns:
    ========
    energy : float
        Energy expectation
    """
    input_list = fix_list(lst, all_param_array=all_param, var_param_array=var_param, fixed_vals_array=fixed_vals)
    energy = cusp_stage3.run_sim_repetitions_stage3(*input_list, bond_length, n_repetitions=n_repetitions,
                                                    exact=no_noise, noisy=gate_error)
    return energy
