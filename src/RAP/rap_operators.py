# File: rap_operators.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Quantum Operators for Rapid Adiabatic Passage (RAP) Simulations.

This module provides specific operator constructions used in RAP 
simulations, focusing on:
- Motional collapse operators (decoherence/cooling).
- Internal two-level transition operators (raising operator).
"""

import numpy as np
from qutip import *


def RAP_collapse_cooling_op(decay_rate, n_motional):
    """
    Constructs the collapse operator for the cooling process in RAP simulations.

    This operator acts on the combined Hilbert space of a 2-level internal 
    system and the motional degree of freedom, representing phonon loss.

    Parameters
    ----------
    decay_rate : float
        The physical decay rate associated with the cooling process.
    n_motional : int
        Dimension of the motional Hilbert space (Fock space cutoff).

    Returns
    -------
    list of Qobj
        A list containing the single collapse operator scaled by the 
        square root of the decay rate.
    """
    op = tensor(
        qeye(2), 
        np.sqrt(decay_rate) * destroy(n_motional)
    )

    return [op]


def RAP_sigmap_2():
    """
    Constructs the raising operator (sigma-plus) for a 2-level system.

    This operator represents the transition from the ground state to the 
    excited state in the internal molecular degree of freedom.

    Returns
    -------
    Qobj
        The 2x2 QuTiP quantum object representing the internal raising operator.
    """
    sigmap_n = np.zeros((2, 2))
    sigmap_n[1, 0] = 1.0

    op_mol = Qobj(sigmap_n)
    
    return op_mol