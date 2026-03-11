# File: odf_operators.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Module defining operators for molecules, motion, and co-trapped ions.

This module provides:
- Sigma operators for three-level molecular systems.
- Decay operators for molecular coherences.
- Position (x) and momentum (p) operators for motion, for:
    - Molecule alone
    - Molecule + ion
    - Molecule with 3-level structure + ion
"""

from qutip import *
import numpy as np


###################################################################################
########################## MOLECULE WITH 3 LEVELS #################################
###################################################################################

# Sigma+ operator: |1> → |0>
sigmap_3 = Qobj(np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0]
]))

# Sigma- operator: |0> → |1>
sigmam_3 = Qobj(np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0]
]))

# Sigma_z operator for |0> and |1> subspace
sigmaz_3 = Qobj(np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 0]
]))


def decay(coherence_time):
    """
    Return the decay (dephasing) operator for a three-level molecule.

    Parameters
    ----------
    coherence_time : float
        Coherence time in microseconds.

    Returns
    -------
    Qobj
        Decay operator scaled by coherence time.
    """
    return np.sqrt(1 / (coherence_time * 2)) * sigmaz_3


def x_op_3mol_atom(N):
    """
    Position operator x for the motional degree of freedom 
    in a system of a 3-level molecule co-trapped with an ion.

    Parameters
    ----------
    N : int
        Dimension of the motional Hilbert space.

    Returns
    -------
    Qobj
        Tensor operator acting on molecule ⊗ motion ⊗ ion.
    """
    return tensor(qeye(3), create(N) + destroy(N), qeye(2)) / 2


def p_op_3mol_atom(N):
    """
    Momentum operator p for the motional degree of freedom 
    in a system of a 3-level molecule co-trapped with an ion.

    Parameters
    ----------
    N : int
        Dimension of the motional Hilbert space.

    Returns
    -------
    Qobj
        Tensor operator acting on molecule ⊗ motion ⊗ ion.
    """
    return tensor(qeye(3), 1j * (create(N) - destroy(N)), qeye(2)) / 2


###################################################################################
########################## MOLECULE + MOTION + ION ################################
###################################################################################


def x_op_mol_atom(N):
    """
    Position operator x for the motional degree of freedom
    in a 2-level molecule co-trapped with an ion.

    Parameters
    ----------
    N : int
        Dimension of the motional Hilbert space.

    Returns
    -------
    Qobj
        Tensor operator acting on molecule ⊗ motion ⊗ ion.
    """
    return tensor(qeye(2), create(N) + destroy(N), qeye(2)) / 2


def p_op_mol_atom(N):
    """
    Momentum operator p for the motional degree of freedom
    in a 2-level molecule co-trapped with an ion.

    Parameters
    ----------
    N : int
        Dimension of the motional Hilbert space.

    Returns
    -------
    Qobj
        Tensor operator acting on molecule ⊗ motion ⊗ ion.
    """
    return tensor(qeye(2), 1j * (create(N) - destroy(N)), qeye(2)) / 2


###################################################################################
########################## MOLECULE + MOTION ######################################
###################################################################################


def x_op_molecule(N):
    """
    Position operator x for the motional degree of freedom
    for a molecule alone.

    Parameters
    ----------
    N : int
        Dimension of the motional Hilbert space.

    Returns
    -------
    Qobj
        Tensor operator acting on molecule ⊗ motion.
    """
    return tensor(qeye(2), create(N) + destroy(N)) / 2


def p_op_molecule(N):
    """
    Momentum operator p for the motional degree of freedom
    for a molecule alone.

    Parameters
    ----------
    N : int
        Dimension of the motional Hilbert space.

    Returns
    -------
    Qobj
        Tensor operator acting on molecule ⊗ motion.
    """
    return tensor(qeye(2), 1j * (create(N) - destroy(N))) / 2