import numpy as np
import os
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import get_sparse_operator, jordan_wigner
from openfermion.utils import expectation, get_ground_state

from cirq.google import XmonQubit


# Load and parse user settings
user_settings = np.load('data/user_settings.npy')
no_sampling_noise = user_settings[0]
gate_error = user_settings[1]
noise_level = user_settings[2]
num_trials = user_settings[3]
bond_lengths = user_settings[4]

# Qubit initialization
q00 = XmonQubit(0, 0)
q01 = XmonQubit(0, 1)
q10 = XmonQubit(1, 0)
q11 = XmonQubit(1, 1)
qubit_ordering = [q11, q10, q01, q00]

# Get molecular data
HDF5_PATH = "h2_sto3g"
BASIS = "sto-3g"
MULTIPLICITY = "singlet"
dist_list = bond_lengths 

hf_energies = []
fci_energies = []
# ground_states = []
# ground_energies = []
# hamiltonians = []
sparse_hamiltonians = []

for dist in dist_list:
    dist = "{0:.1f}".format(dist)
    file_path = os.path.join(HDF5_PATH, "H2_{0}_{1}_{2}.hdf5".format(BASIS, MULTIPLICITY, dist))

    molecule = MolecularData(filename=file_path)
    n_qubits = molecule.n_qubits

    hf_energies.append(molecule.hf_energy)
    fci_energies.append(molecule.fci_energy)

    molecular_ham = molecule.get_molecular_hamiltonian()
    #hamiltonians.append(molecular_ham)
    molecular_ham_sparse = get_sparse_operator(operator=molecular_ham,
                                               n_qubits=n_qubits)
    sparse_hamiltonians.append(molecular_ham_sparse)
    
    # ground_energy, ground_state = get_ground_state(molecular_ham_sparse)
    # ground_energies.append(ground_energy)
    # ground_states.append(ground_state)


def compute_energy_expectation(bond_length, state):
    """Computes the energy expectation given the bond length (to fetch the
    Hamiltonian) and state.

    Args:
    =====
    bond_length : float
        Bond length for fetching correct Hamiltonian
    state : numpy.ndarray
        State vector

    Returns: 
    ========
    energy : float
        Energy expectation value

    """
    energy = expectation(sparse_hamiltonians[dist_list.index(bond_length)], state)
    return np.real(energy)

def fetch_ground_energy(bond_length):
    """Fetches exact ground state energy given the bond length.

    Args:
    =====
    bond_length : float
        Bond length for fetching correct ground state energy

    Returns:
    ========
    exact_energy : float
        Exact ground state energy
    """
    exact_energy = fci_energies[dist_list.index(bond_length)]
    return exact_energy
