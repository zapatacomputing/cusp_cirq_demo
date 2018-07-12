
"""Routines for Stage Two of CUSP: Training the Quantum Autoencoder."""

import numpy as np

from cirq import Circuit, MeasurementGate, ParamResolver
from cirq.ops import *
from cirq.google import ExpZGate, XmonQubit, XmonSimulator
from cirq.circuits import InsertStrategy
from cirq.contrib.jobs.job import Job
from cirq.contrib.jobs.depolarizer_channel import DepolarizerChannel

import settings
from cusp_demo_utils import *


# Probability of noisy Z rotations
noise_level = settings.noise_level

# Qubit initialization
q00, q01, q10, q11 = settings.q00, settings.q01, settings.q10, settings.q11 
qubit_ordering = settings.qubit_ordering

def _input_prep_gates_stage2(alpha):
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
                        X2(q10),
                        X2(q11)])
    return state_prep_gates

def compression_circuit(a, b, x, z, alpha, exact=False):
    """Returns compression circuit (state preparation circuit followed by
    encoding circuit).

    Args:
    =====
    a, b, x, z : numeric
        Circuit parameters for encoding circuit
    alpha : numeric
        Parameter for state preparation circuit
    exact : bool
        If True, works with wavefunction

    Returns:
    ========
    comp_circuit : cirq.Circuit
        Compression circuit
    """
    comp_circuit = Circuit()
    comp_circuit.append(_input_prep_gates_stage2(alpha))
    comp_circuit.append(param_CNOT(a, b, x, z, q01, q00))
    comp_circuit.append(param_CNOT(a, b, x, z, q11, q10), strategy=InsertStrategy.EARLIEST)
    comp_circuit.append(param_CNOT(a, b, x, z, q11, q01))
    if exact == False:
        comp_circuit.append([MeasurementGate('r00').on(q00),
                            MeasurementGate('r01').on(q01),
                            MeasurementGate('r10').on(q10),
                            MeasurementGate('r11').on(q11)])
    return comp_circuit

def noisy_job(a, b, x, z, alpha, exact=False):
    """Adds noise to compression circuit.

    Args:
    =====
    a, b, x, z : numeric
        Circuit parameters for encoding circuit
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
    job = Job(compression_circuit(a, b, x, z, alpha, exact))
    noisy = DepolarizerChannel(probability=noise_level)
    noisy_job = noisy.transform_job(job)
    param_resolvers = [ParamResolver({k:v for k, v in e}) for e in noisy_job.sweep.param_tuples()]
    return noisy_job.circuit, param_resolvers

def _run_sim_stage2(a, b, x, z, alpha, exact=False, print_circuit=False, noisy=False):
    """Executes circuit a single time. Outputs 1 for a success (i.e. reference qubits are |000>)
    and 0 for a failure.

    Args:
    =====
    a, b, x, z : numeric
        Circuit parameters for encoding circuit
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
    total : int
        Value of 1 if reference qubits are all 0's. Value of 0 else.
    """
    simulator = XmonSimulator()

    if noisy:
        circuit_run, resolvers = noisy_job(a, b, x, z, alpha, exact)
    else:
        circuit_run = compression_circuit(a, b, x, z, alpha, exact)
    
    if exact:
        if noisy:
            for resolver in resolvers:
                result = simulator.simulate(circuit=circuit_run, param_resolver=resolver)
        else:
            result = simulator.simulate(circuit=circuit_run)
        avg = 0
        for j in range(2):
            avg += np.abs(result.final_state[j])**2
        return avg
    
    else:
        if noisy:
            for resolver in resolvers:
                result = simulator.run(circuit=circuit_run,
                                       param_resolver=resolver, repetitions=1)
        else:
            result = simulator.run(circuit=circuit_run, repetitions=1)
    
    reference_measurements = []
    reference_labels = ['r00', 'r01', 'r10']
    for j in reference_labels:
        reference_measurements.append(int(result.measurements[j][0]))
    total = 0
    res = []
    for y in range(3):
        res.append(reference_measurements[y])
    if res == [0, 0, 0]:
        total = 1
    if print_circuit==True:
        print(circuit_run.to_text_diagram(use_unicode_characters=False))
    return total

def compute_stage2_cost_function(a, b, x, z, alpha, n_repetitions, exact=False, noisy=False):
    """Executes circuit multiple times and computes the average fidelity.
    over n times (n_repetitions).

    Args:
    =====
    a, b, x, z : numeric
        Circuit parameters for encoding circuit
    alpha : numeric
        Parameter for state preparation circuit
    n_repetitions : int
        Number of circuit runs
    exact : bool
        If True, works with wavefunction
    noisy : bool
        If True, runs noisy version of circuit

    Returns:
    ========
    avg_fid : float
        Average fidelity (maximum: 1)
    """
    if exact == True and noisy == False:
        return _run_sim_stage2(a, b, x, z, alpha, exact=exact, print_circuit=False, noisy=noisy)
    
    success_count = 0
    for k in range(n_repetitions):
        success_count += _run_sim_stage2(a, b, x, z, alpha, exact=exact, print_circuit=False, noisy=noisy)

    avg_fid = float(success_count) / float(n_repetitions)
    return avg_fid
