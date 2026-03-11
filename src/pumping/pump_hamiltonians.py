"""
Module for constructing Blue Sideband (BSB) Hamiltonians.

This module provides functions to build time-dependent Hamiltonians for 
molecular transitions coupled to a motional mode. It includes tools for 
individual 2-level transitions, summations of multiple processes, and 
automated generation of Hamiltonian arguments from transition dataframes.
"""

from qutip import *
from pumping.pump_operators import sigmap_n
import numpy as np
from typing import Any, Dict, List, Tuple, Callable, Optional


def H_bsb_2levels(t: float, args: Dict[str, Any]) -> Qobj:
    """
    Constructs the Hamiltonian for a BSB process between two specific levels.

    The Hamiltonian represents a transition in the interaction picture, 
    coupling the internal state transition to the creation/annihilation of 
    phonons in the motional mode.

    Parameters
    ----------
    t : float
        The current time step in the simulation.
    args : dict
        Dictionary containing the following physical parameters:
        - 'n_motional' (int): Dimension of the motional Fock space.
        - 'n_internal' (int): Dimension of the internal state space.
        - 'initial' (int): Index of the starting molecular state.
        - 'final' (int): Index of the target molecular state.
        - 'rabi_rate' (float): Reduced Rabi rate for the transition.
        - 'coupling' (float): Coupling strength.
        - 'w_mol' (float): Transition frequency of the molecule.
        - 'laser_detuning' (float): Detuning of the driving laser.

    Returns
    -------
    Qobj
        The time-dependent Hamiltonian operator for the single transition.
    """
    n_motional = args['n_motional']
    n_internal = args['n_internal']
    initial = args['initial']
    final = args['final']
    rabi_rate = args['rabi_rate']
    coupling = args['coupling']
    w_mol = args['w_mol']

    laser_detuning = args['laser_detuning']
    det_mol_trans = laser_detuning - w_mol

    sigma_plus_mol = tensor(sigmap_n(initial, final, n_internal), qeye(n_motional))
    a = tensor(qeye(n_internal), destroy(n_motional))

    H_term = sigma_plus_mol * a.dag() * np.exp(-1j * det_mol_trans * t)
    
    return rabi_rate * np.abs(coupling) / 2 * (H_term + H_term.dag())


def H_bsb_total(t: float, args: Dict[str, Any]) -> Qobj:
    """
    Computes the total time-dependent BSB Hamiltonian as a sum of multiple terms.

    This function iterates through a list of individual transition arguments
    and aggregates their respective Hamiltonians.

    Parameters
    ----------
    t : float
        Current time step.
    args : dict
        A dictionary containing a 'terms' key, which is a list of dictionaries,
        each formatted as the 'args' for H_bsb_2levels.

    Returns
    -------
    H : Qobj
        The total sum Hamiltonian at time t.
    """
    H = 0
    for term_args in args['terms']:
        H += H_bsb_2levels(t, term_args)
    return H


def H_bsb_manifold(
    dataframe: Any,
    j_val: int,
    is_minus: bool,
    n_motional: int,
    n_internal: int,
    rabi_rate: float,
    laser_detuning: float,
    manifold: Optional[str] = None
) -> Tuple[Callable, Dict[str, List[Dict[str, Any]]]]:
    """
    Generates the total BSB Hamiltonian and its arguments from a transition dataset.

    This function maps global transition indices from a dataframe into local 
    indices based on the selected rotational manifold (j) and transition type.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing transition indices, couplings, and energy differences.
    j_val : int
        The total angular momentum value characterizing the manifold.
    is_minus : bool
        Direction of the transition (determines initial/final state mapping).
    n_motional : int
        Number of Fock states in the motional mode.
    n_internal : int
        Dimension of the internal state space.
    rabi_rate : float
        Reduced Rabi rate applied to the manifold.
    laser_detuning : float
        Detuning of the laser frequency.
    manifold : str, optional
        Selection of the manifold ('upper', 'lower', or 'all'). Default is None.

    Returns
    -------
    H_tot : function
        The H_bsb_total function to be used in QuTiP solvers (mesolve/sesolve).
    args : dict
        The parameters required by the H_tot function.

    Raises
    ------
    ValueError
        If an invalid manifold name is provided.
    """
    H_terms_args = []

    if manifold is None or manifold == "upper" or manifold == "all":
        rescaled_index = int(np.sum([2 * (2 * j + 1) for j in range(j_val)]))
    elif manifold == "lower":
        rescaled_index = int(np.sum([2 * (2 * j + 1) for j in range(j_val)]) + (2 * j_val + 1))
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

        if is_minus:
            w_mol = row["energy_diff"] * 1e-3
        else:
            w_mol = -row["energy_diff"] * 1e-3

        term_args = {
            'n_motional': int(n_motional), 
            'n_internal': int(n_internal), 
            'initial': int(initial), 
            'final': int(final), 
            'rabi_rate': rabi_rate,
            'coupling': np.abs(coupling), 
            'w_mol': 2 * np.pi * w_mol, 
            'laser_detuning': laser_detuning
        }
        
        H_terms_args.append(term_args)
        
    H_tot = H_bsb_total
    args = {'terms': H_terms_args}

    return H_tot, args