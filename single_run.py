import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from cirq import Circuit, MeasurementGate, ParamResolver
from cirq.ops import *
from cirq.google import ExpWGate, ExpZGate, XmonSimulator
from cirq.circuits import InsertStrategy
from cirq.contrib.jobs.job import Job
from cirq.contrib.jobs.depolarizer_channel import DepolarizerChannel

import sys
from config import CODE_DIRECTORY
sys.path.append(CODE_DIRECTORY)
import settings
import cusp_stage1

#Repetitions
n_repetitions = 20000

#Noise Level
noise_level = 0.002

#Bond Length
bond_length = 2.5

#Stage 1 Settings
alpha = 0.43960449

#Stage 2 Settings
a = 0.75003402  # half_turns on W gate, ideal is .25
b = -0.09734939  # axis_half_turns on W gate, ideal is .5
x = 0.99984087  # half_turns on 11 gate, ideal is 1
z = 1.00002832  # half_turns on Z gate, ideal is 1

#Stage 3 Settings
aht = 0.22394839066930666
ht = 0.4394920830043302
zz = 0.06873740782090981

from cusp_demo_utils import *

# Qubit initialization
q00, q01, q10, q11 = settings.q00, settings.q01, settings.q10, settings.q11
qubit_ordering = settings.qubit_ordering

def _latent_space_circuit_gates(aht, ht, zz):
    """Helper routine for producing sequence of gates
    for the latent space circuit.

    Args:
    =====
    aht, ht, zz : numeric
        Parameters for latent space circuit

    Returns:
    ========
    state_prep_gates : list
        List (ordered sequence) of Cirq gates for the latent space circuit
    """
    input_param_W = ExpWGate(half_turns=ht, axis_half_turns=aht)
    input_param_Z = ExpZGate(half_turns=zz)
    circuit = [input_param_W(q11), input_param_Z(q11)]
    return circuit

def decoder_circuit(aht, ht, zz, exact=False):
    """Returns latent space circuit followed by decoding circuit.

    Args:
    =====
    aht, ht, zz : numeric
        Parameters for latent space circuit
    exact : bool
        If True, works with wavefunction

    Returns:
    ========
    dc_circuit : cirq.Circuit
        Decoding circuit
    """
    dc_circuit = Circuit()
    dc_circuit.append(_latent_space_circuit_gates(aht, ht, zz))
    dc_circuit.append(param_CNOT(a, b, x, z, q11, q01))
    dc_circuit.append(param_CNOT(a, b, x, z, q01, q00))
    dc_circuit.append(param_CNOT(a, b, x, z, q11, q10), strategy=InsertStrategy.EARLIEST)
    dc_circuit.append([X(q11), X(q10)])
    if exact is False:
        dc_circuit.append([MeasurementGate('r00').on(q00),
                          MeasurementGate('r01').on(q01),
                          MeasurementGate('r10').on(q10),
                          MeasurementGate('r11').on(q11)])
    return dc_circuit

def noisy_job_stage3(aht, ht, zz, exact=False):
    """Adds noise to decoding circuit.

    Args:
    =====
    aht, ht, zz : numeric
        Circuit parameters for decoding circuit
    exact : bool
        If True, works with wavefunction

    Returns:
    ========
    noisy_circuit : cirq.Circuit
        Noisy version of input circuit
    param_resolvers : list
    """
    job = Job(decoder_circuit(aht, ht, zz, exact))
    noisy = DepolarizerChannel(probability=noise_level)
    noisy_job = noisy.transform_job(job)
    param_resolvers = [ParamResolver({k:v for k, v in e})
             for e in noisy_job.sweep.param_tuples()]
    return noisy_job.circuit, param_resolvers

def _run_sim_stage3(aht, ht, zz, exact=False, print_circuit=False, noisy=False):
    """Helper routine to executes state preparation circuit a single time.
    Outputs a state vector.

    Args:
    =====
    aht, ht, zz : numeric
        Parameters for decoding circuit
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
    
    if noisy:
        circuit_run, resolvers = noisy_job_stage3(aht, ht, zz, exact)
        for resolver in resolvers:
            result = simulator.simulate(circuit=circuit_run, param_resolver=resolver,
                                        qubit_order=qubit_ordering)
    else:
        circuit_run = decoder_circuit(aht, ht, zz, exact)
        result = simulator.simulate(circuit=circuit_run, qubit_order=qubit_ordering)
    
    if print_circuit:
        print(circuit_run.to_text_diagram(use_unicode_characters=False))
    return result.final_state

def one_run(aht, ht, zz, bond_length):
    f_state = _run_sim_stage3(aht=aht, ht=ht, zz=zz, exact=True, print_circuit=False, noisy=True)
    return settings.compute_energy_expectation(bond_length, particle_number_conserve(f_state))

def run_sim_repetitions_stage3(aht, ht, zz, bond_length, n_repetitions, exact=True, noisy=False):
    """Executes state preparation circuit multiple times and computes the energy expectation
    over n times (n_repetitions).

    Args:
    =====
    aht, ht, zz : numeric
        Parameters for decoding circuit
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
    if exact == True and noisy == False:
        final_state = _run_sim_stage3(aht, ht, zz, exact=True, print_circuit=False, noisy=False)
        energy_expectation = settings.compute_energy_expectation(bond_length, particle_number_conserve(final_state))
        return energy_expectation
    
   # energy_expectation = 0
   # for k in range(n_repetitions):
   #     energy_expectation += one_run(aht,ht,zz,bond_length)
   # energy_expectation = energy_expectation / float(n_repetitions)

    p = Pool()
    args = [(aht, ht, zz, bond_length)] * n_repetitions
    results = p.starmap(one_run,args)
    energy_expectation = np.array(results).mean()
    return energy_expectation

energy1 = cusp_stage1.compute_stage1_cost_function(alpha, bond_length, n_repetitions, exact=True, noisy=True)
print('VQE Energy: '+str(energy1))

energy3 = run_sim_repetitions_stage3(aht, ht, zz, bond_length, n_repetitions, exact=True, noisy=True)
print('CUSP Energy: '+str(energy3))
