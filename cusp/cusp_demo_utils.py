
"""Utility functions for CUSP demo."""

import numpy as np

from cirq import ParamResolver
from cirq.contrib.jobs.job import Job
from cirq.contrib.jobs.depolarizer_channel import DepolarizerChannel
from cirq.google import Exp11Gate, ExpWGate, ExpZGate, XmonSimulator
import cirq.ops as op


X2 = op.X**0.5
X2inv = op.X**-0.5

def param_W(i, j):
    """Returns parametrized ExpWGate."""
    return ExpWGate(half_turns=i, axis_half_turns=j)

def param_Winv(i,j):
    """Returns parametrized inverse of ExpWGate."""
    return ExpWGate(half_turns=-i, axis_half_turns=j)

def param_11(i):
    """Returns parametrized Exp11Gate."""
    return Exp11Gate(half_turns=i)

def param_Z(i):
    """Returns parametrized ExpZGate."""
    return ExpZGate(half_turns=i)

def param_H(w_half_turns, w_axis_half_turns, z_half_turns, target):
    """Returns a parametrized Hadamard gate
    composed of gates in Cirq's native gate set."""
    op = []
    op.append(param_Winv(w_half_turns, w_axis_half_turns)(target))
    op.append(param_Z(z_half_turns)(target))
    op.append(param_W(w_half_turns, w_axis_half_turns)(target))
    return op

def param_CNOT(w_half_turns, w_axis_half_turns, proj11_half_turns, z_half_turns, control,target):
    """Returns a parametrized Hadamard gate
    composed of gates in Cirq's native gate set."""
    op = []
    op.append(param_H(w_half_turns, w_axis_half_turns, z_half_turns, target))
    op.append(param_11(proj11_half_turns)(control, target))
    op.append(param_H(w_half_turns, w_axis_half_turns, z_half_turns, target))
    out = np.asarray(op)
    out.flatten()
    out.tolist()
    return out

# A wrapper to make sure that the selection of variational parameters is properly resolved when input to the function
def fix_list(lst, all_param_array, var_param_array, fixed_vals_array):
    out_list = []
    count = 0
    if type(lst)!=np.ndarray:
        lst = [lst]
    for j in all_param_array:
        if j in var_param_array:
            out_list.append(lst[count])
            count = count + 1
        else:
            out_list.append(fixed_vals_array[all_param_array.index(j)])
    return out_list

def particle_number_conserve(state_vector):
    """Removes all amplitude over states that do not conserve
    the number of partices in the given state.

    Args:
    =====
    state_vector : list or numpy.ndarray
        State vector
    
    Returns:
    ========
    state_vector : list or numpy.ndarray
        State vector with conserved particle number
    """
    # NOTE: hard-coded
    non_particle_conserved_indices = [0, 1, 2, 4, 7, 8, 11, 13, 14, 15]
    
    for i in non_particle_conserved_indices:
        state_vector[i] = 0
    return state_vector

def add_noise(circuit, noise_level):
    """Adds depolarizing noise to circuit.

    Args:
    =====
    circuit : cirq.Circuit
        Circuit in which noise to be added
    noise_level : float
        Probability of a qubit being affected by the noise channel in a given moment

    Returns: 
    ========
    noisy_circuit : cirq.Circuit
        Noisy version of circuit
    """
    job = Job(circuit)
    noisy_channel = DepolarizerChannel(probability=noise_level)
    noisy_job = noisy_channel.transform_job(job)
    param_resolvers = [ParamResolver({k:v for k,v in e})
             for e in noisy_job.sweep.param_tuples()]
    return noisy_job.circuit, param_resolvers
