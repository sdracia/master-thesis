from qutip import *
from pumping.pump_operators import sigmap_n
import numpy as np



def H_bsb_2levels(t, args) :
    """
    Hamiltonian for the BSB process occurring between only two levels of the molecule.
    The dimensionality of the Hamitonian is n_internal * n_motional, however the BSB transition involves only two states in the internal space.
    """
    n_motional = args['n_motional']     # number of motional states
    n_internal = args['n_internal']   # number of internal states
    initial = args['initial']       # index of the initial state
    final = args['final']           # index of the final state
    rabi_rate = args['rabi_rate']   # Rabi rate of the transition
    coupling = args['coupling']     # coupling constant of the transition
    w_mol = args['w_mol']           # molecular transition frequency

    laser_detuning = args['laser_detuning']     # laser detuning
    det_mol_trans = laser_detuning - w_mol      # detuning of the molecular transition

    sigma_plus_mol = tensor(sigmap_n(initial, final, n_internal), qeye(n_motional))
    a = tensor(qeye(n_internal), destroy(n_motional))

    H_term = sigma_plus_mol * a.dag() * np.exp(-1j*det_mol_trans*t)
    
    return rabi_rate * np.abs(coupling) /2 * (H_term + H_term.dag())



def H_bsb_total(t, args):
    """
    Total Hamiltonian for the BSB process.
    The Hamiltonian is a sum of the Hamiltonians for each transition.
    Each term in the sum is a Hamiltonian for a BSB process occurring between two levels of the molecule, having its own arguments
    """
    H = 0
    for term_args in args['terms']:
        H += H_bsb_2levels(t, term_args)
    return H




def H_bsb_manifold(dataframe, j_val, is_minus, n_motional, n_internal, rabi_rate, laser_detuning, manifold = None):
    """
    Function for computing the Hamiltonian for the BSB process, based on the dataframe containing the information about the transitions.
    Parameters:
    - dataframe: pandas dataframe containing the information about the transitions
    - j_val: value of the total angular momentum of the molecule
    - is_minus: boolean value indicating if the transition is from a lower level to a higher level or vice versa
    - n_motional: number of motional states
    - n_internal: number of internal states
    - rabi_rate: Rabi rate of the transition
    - laser_detuning: laser detuning
    """

    H_tot = []
    H_terms_args = []

    if manifold == None or manifold == "upper" or manifold == "all":
        rescaled_index = int(np.sum([2*(2*j+1) for j in range(j_val)]))  # the indices are rescaled to start from 0
    elif manifold == "lower":
        rescaled_index = int(np.sum([2*(2*j+1) for j in range(j_val)]) + (2*j_val+1))
    else:
        raise ValueError("Manifold must be 'upper' or 'lower'")

    for _, row in dataframe.iterrows():

        if is_minus:
            initial = int(row["index1"]) - rescaled_index
            final = int(row["index2"]) - rescaled_index
        else:
            initial = int(row["index2"]) - rescaled_index
            final = int(row["index1"]) - rescaled_index

        coupling = row["coupling"]

        # since the detuning are computed wrt is_minus = True, then if is_minus = False, the values of the detunings are inverted
        if is_minus:
            w_mol = row["energy_diff"] * 1e-3
        else:
            w_mol = -row["energy_diff"] * 1e-3

        term_args = {'n_motional': int(n_motional), 
                'n_internal': int(n_internal), 
                'initial': int(initial), 
                'final': int(final), 
                'rabi_rate': rabi_rate,
                'coupling': np.abs(coupling), 
                'w_mol': 2* np.pi * w_mol, 
                'laser_detuning': laser_detuning}
        

        H_terms_args.append(term_args)
        

    H_tot = H_bsb_total     # H_tot is the total Hamiltonian of the system. It will be the sum of all the terms of the Hamiltonian
    args = {'terms': H_terms_args}

    return H_tot, args



