# File: pump_operators.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Quantum Operators Module for Molecular Cooling and State Manipulation.

This module provides utility functions to construct specific quantum operators 
using QuTiP, focusing on:
- Internal state transition operators (sigmap).
- Collapse operators for motional cooling simulations.
"""

import numpy as np
from qutip import *
from typing import List


def sigmap_n(
    init_index: int, 
    final_index: int, 
    n_internal: int
) -> Qobj:
    """
    Constructs the raising/transition operator (sigmap) in the internal state space.

    The operator is represented as a matrix of size (n_internal x n_internal) 
    with a single non-zero entry corresponding to the transition between 
    the specified indices.

    Parameters
    ----------
    init_index : int
        The index of the initial state in the internal space.
    final_index : int
        The index of the final state in the internal space.
    n_internal : int
        The total number of internal states (dimension of the Hilbert space).

    Returns
    -------
    op_mol : Qobj
        The QuTiP quantum object representing the transition operator.
    """
    # Initialize a zero matrix of the required internal dimension
    sigmap_n_matrix = np.zeros((n_internal, n_internal))
    
    # Set the transition element from init_index to final_index
    sigmap_n_matrix[final_index, init_index] = 1.0

    # Convert the numpy array to a QuTiP quantum object
    op_mol = Qobj(sigmap_n_matrix)
    
    return op_mol


def collapse_cooling_op(
    decay_rate: float, 
    n_internal: int, 
    n_motional: int
) -> List[Qobj]:
    """
    Constructs the collapse operator used for the motional cooling process.

    The operator is defined as the tensor product of the identity operator in 
    the internal molecular space and the lowering operator in the motional 
    space, scaled by the square root of the decay rate.

    Parameters
    ----------
    decay_rate : float
        The physical decay rate (Gamma) associated with the cooling process.
    n_internal : int
        The dimension of the internal molecular state space.
    n_motional : int
        The dimension of the motional state space (Fock space cutoff).

    Returns
    -------
    list of Qobj
        A list containing the QuTiP collapse operator as required by solver functions.
    """
    op = tensor(
        qeye(n_internal), 
        np.sqrt(decay_rate) * destroy(n_motional)
    )

    return [op]