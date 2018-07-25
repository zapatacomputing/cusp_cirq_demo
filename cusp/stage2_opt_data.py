
"""Driver routine for Stage Two of CUSP: Training the Quantum Autoencoder."""

import cusp_stage2
import numpy as np
from scipy.optimize import minimize

import settings
from set_settings import *
from cusp_demo_utils import *

# Settings for running the QAE optimization
user_parameters_stage2 = np.load('data/user_parameters_stage2.npy')
#w1 = 'w1'
#w2 = 'w2'
#z = 'z'
#cz = 'cz'
var_param = user_parameters_stage2[0]

num_trials = settings.num_trials
bond_lengths = settings.bond_lengths
no_noise = settings.no_sampling_noise
gate_error = settings.gate_error

# A wrapper to make sure that the selection of var_param above is properly resolved when input to the function
num_param = len(var_param)
fixed_vals = [user_parameters_stage2[1], user_parameters_stage2[2], user_parameters_stage2[3], user_parameters_stage2[4]]
all_param = ['w1','w2','z','cz']
result_list = []

def compute_avg_fid_proxy(params, training_states, n_repetitions, exact=no_noise, noisy=gate_error):
    """Computes cost function value, 1 - average fidelity, over the training set.
    If the cost function is 0, the QAE is perfectly encoding. Max error is 1.

    Args:
    =====
    params : numpy.ndarray
        Vector of QAE parameters
    training_states : list
        List of training state parameters
    n_repetitions : int
        Number of circuit runs/trials
    exact : bool
        If True, works with wavefunction
    noisy : bool
        If True, runs noisy version of circuit

    Returns:
    ========
    cost_fcn_val : float
        Cost function value equivalent to 1 - average fidelity
    """
    input_list = fix_list(params, all_param_array=all_param, var_param_array=var_param, fixed_vals_array=fixed_vals)
    fidelities = []
    for training_state in training_states:
        fid = cusp_stage2.compute_stage2_cost_function(*input_list, alpha=training_state, n_repetitions=n_repetitions,
                                                       exact=exact, noisy=noisy)
        fidelities.append(fid)
    avg_fid = np.mean(fidelities)
    global result_list
    result_list.append(1-avg_fid)
    print(1-avg_fid)
    return 1. - avg_fid

def run_qae_optimization(training_states, n_repetitions, exact=no_noise, noisy=gate_error):
    """Runs optimization for QAE.

    Args:
    =====
    training_states : list[float]
        List of optimized training state parameters
    n_repetitions : int
        Number of circuit runs/trials
    exact : bool
        If True, works with wavefunction
    noisy : bool
        If True, runs noisy version of circuit

    Returns:
    ========
    optimized_qae_params : numpy.ndarray
        Vector of optimized QAE circuit parameters
    """
    global result_list
    # Initialize parameters
    half_turn_min = 0
    half_turn_max = 2
    init_params = np.random.uniform(low=half_turn_min, high=half_turn_max,
                                    size=num_param)

    # Optimization using Nelder-Mead.
    h2_qae_wrap = lambda params: compute_avg_fid_proxy(params, training_states=training_states,
                                                       n_repetitions=n_repetitions, exact=exact, noisy=noisy)
    
    if noisy:
        maxiter = 200
    else:
        maxiter = None
        
    res = minimize(h2_qae_wrap, init_params, args=(),
                   method='Nelder-Mead', tol=None, 
                   options={'disp': False, 'maxiter': maxiter, 'xatol': 0.001,
                   'return_all': False, 'fatol': 0.001})
    np.savetxt('stage2_data.csv',result_list, delimiter=',')
    return res.x
