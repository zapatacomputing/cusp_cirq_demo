
"""Routines for Stage One of CUSP: Training Set Preparation."""

import numpy as np
from multiprocessing import Pool

from cirq import Circuit, MeasurementGate, ParamResolver
from cirq.ops import *
from cirq.google import Exp11Gate, ExpWGate, ExpZGate, XmonQubit, XmonSimulator
from cirq.contrib.jobs.job import Job
from cirq.contrib.jobs.depolarizer_channel import DepolarizerChannel

import settings
from cusp_demo_utils import *


# Adjusts the probability of noisy Z rotations occuring in the circuit
noise_level = settings.noise_level

# Qubit initialization
q00, q01, q10, q11 = settings.q00, settings.q01, settings.q10, settings.q11
qubit_ordering = settings.qubit_ordering

def _input_prep_gates(alpha):
    """Helper routine for producing sequence of gates
    for the state preparation circuit.

    Args:
    =====
    alpha : numeric
        Parameter for state preparation circuit

    Returns:
    ========
    state_prep_gates : list
        List (ordered sequence) of Cirq gates for the state preparation circuit

    """
    state_prep_gates = ([X(q10),
                        X(q11),
                        H(q00),
                        X2(q01),
                        X2(q10),
                        X2(q11),
                        CNOT(q00, q01),
                        CNOT(q01, q10),
                        CNOT(q10, q11),
                        param_Z(alpha)(q11),
                        CNOT(q10, q11),
                        CNOT(q01, q10),
                        CNOT(q00, q01),
                        H(q00),
                        X2inv(q01),
                        X2inv(q10),
                        X2inv(q11)])
    return state_prep_gates

def state_prep_circuit(alpha, exact=False):
    """Returns state preparation circuit.

    Args:
    =====
    alpha : numeric
        Parameter for state preparation circuit
    exact : bool
        If True, works with wavefunction

    Returns:
    ========
    sp_circuit : cirq.Circuit
        State preparation circuit
    """
    sp_circuit = Circuit()
    sp_circuit.append(_input_prep_gates(alpha))
    if exact is False:
        sp_circuit.append([MeasurementGate('r00').on(q00),
                          MeasurementGate('r01').on(q01),
                          MeasurementGate('r10').on(q10),
                          MeasurementGate('r11').on(q11)])
    return sp_circuit

def noisy_job_stage1(alpha, exact=False):
    """Adds noise to state preparation circuit.

    Args:
    =====
    alpha : numeric
        Parameter for state preparation circuit
    exact : bool
        If True, works with wavefunction

    Returns:
    ========
    noisy_circuit : cirq.Circuit
        Noisy version of input circuit
    param_resolvers : list
    """
    job = Job(state_prep_circuit(alpha, exact))
    noisy = DepolarizerChannel(probability=noise_level)
    noisy_job = noisy.transform_job(job)
    param_resolvers = ([ParamResolver({k:v for k,v in e})
                       for e in noisy_job.sweep.param_tuples()])
    return noisy_job.circuit, param_resolvers

def _run_sim_stage1(alpha, exact=True, print_circuit=False, noisy=False):
    """Helper routine to executes state preparation circuit a single time.
    Outputs a state vector of the circuit.

    Args:
    =====
    alpha : numeric
        Parameter for state preparation circuit
    exact : bool
        If True, works with wavefunction
    print_circuit : bool
        If True, prints circuit
    noisy : bool
        If True, runs noisy version of circuit

    Returns:
    ========
    final_state : numpy.ndarray
        Final state vector
    """
    exact = True # NOTE: Hard-coded for now
    simulator = XmonSimulator()
    
    # Set up and execute circuit
    if noisy:
        circuit_run, resolvers = noisy_job_stage1(alpha, exact)
        for resolver in resolvers:
            result = simulator.simulate(circuit=circuit_run, param_resolver=resolver,
                                        qubit_order=qubit_ordering)
    else:
        circuit_run = state_prep_circuit(alpha, exact)
        result = simulator.simulate(circuit=circuit_run, qubit_order=qubit_ordering)

    if print_circuit:
        print(circuit_run.to_text_diagram(use_unicode_characters=False))
    return result.final_state

def one_run(alpha, bond_length):
    f_state = _run_sim_stage1(alpha=alpha, exact=True, print_circuit=False, noisy=True)
    return settings.compute_energy_expectation(bond_length, particle_number_conserve(f_state))
                       
def compute_stage1_cost_function(alpha, bond_length, n_repetitions=100, exact=True, noisy=False):
    """Executes state preparation circuit multiple times and computes the energy expectation
    over n times (n_repetitions).

    Args:
    =====
    alpha : numeric
        Parameter for state preparation circuit
    bond_length : float
        Bond length
    n_repetitions : int
        Number of circuit runs
    exact : bool
        If True, works with wavefunction
    noisy : bool
        If True, runs noisy version of circuit

    Returns:
    ========
    energy_expectation : float
        Energy expectation value
    """
    if exact is True and noisy is False:
        f_state = _run_sim_stage1(alpha=alpha, exact=exact, print_circuit=False, noisy=noisy)
        energy_expectation = settings.compute_energy_expectation(bond_length, particle_number_conserve(f_state))
        return energy_expectation
    
    energy_expectation = 0
    
    p = Pool()
    args = [(alpha, bond_length)] * n_repetitions
    results = p.starmap(one_run,args)
    energy_expectation = np.array(results).mean()
    p.close()
    return energy_expectation
