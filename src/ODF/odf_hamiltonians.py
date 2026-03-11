"""
Module defining time-dependent Hamiltonians for molecules, motion, and co-trapped ions.

This module provides:
- Optical dipole force (ODF) Hamiltonians.
- Blue sideband (BSB) and red sideband (RSB) Hamiltonians.
- Combined Hamiltonians (BSB + ODF).
- Implementations for:
    - Three-level molecule + ion
    - Two-level molecule + ion
    - Molecule alone with motion
"""

from qutip import *
import numpy as np
from ODF.odf_operators import sigmap_3, sigmam_3

###################################################################################
########################## 3-LEVEL MOLECULE + ION #################################
###################################################################################

def H_mol3_atom_odf(t, args):
    """
    Optical dipole force (ODF) Hamiltonian for a 3-level molecule co-trapped with an ion.

    t : float, time in μs
    args : dict
        N : int, motional Hilbert space dimension
        rabi_rate : float, Rabi frequency for ODF
        d : float, ODF detuning frequency

    Returns
    -------
    Qobj : Time-dependent ODF Hamiltonian
    """
    N = args['N']
    x = tensor(qeye(3), create(N) + destroy(N), qeye(2)) / 2
    p = tensor(qeye(3), 1j * (create(N) - destroy(N)), qeye(2)) / 2

    rabi_rate = args['rabi_rate']
    d = args['d']

    return x * rabi_rate * np.sin(d * t) + p * rabi_rate * np.cos(d * t)


def H_mol3_atom_bsb(t, args):
    """
    Blue sideband (BSB) Hamiltonian for a 3-level molecule co-trapped with an ion.

    t : float
    args : dict
        N : int, motional Hilbert space dimension
        rabi_rate_molecule : float
        d : float, detuning
        w_mol : float, transition frequency of molecule

    Returns
    -------
    Qobj : Time-dependent BSB Hamiltonian
    """
    N = args['N']
    sigma_plus_mol = tensor(sigmap_3, qeye(N), qeye(2))
    sigma_minus_mol = tensor(sigmam_3, qeye(N), qeye(2))
    a = tensor(qeye(3), destroy(N), qeye(2))

    rabi_rate_molecule = args['rabi_rate_molecule']
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol

    return rabi_rate_molecule / 2 * (sigma_plus_mol * a.dag() * np.exp(-1j * d_mol * t) +
                                     sigma_minus_mol * a * np.exp(1j * d_mol * t))


def H_mol3_atom_tot(t, args):
    """
    Combined BSB + ODF Hamiltonian for a 3-level molecule co-trapped with an ion.

    t : float
    args : dict
        N : int, motional Hilbert space dimension
        rabi_rate : float, ODF Rabi frequency
        rabi_rate_molecule : float, BSB Rabi frequency
        d : float, detuning
        w_mol : float, transition frequency of molecule

    Returns
    -------
    Qobj : Total Hamiltonian (ODF + BSB)
    """
    N = args['N']
    sigma_plus_mol = tensor(sigmap_3, qeye(N), qeye(2))
    sigma_minus_mol = tensor(sigmam_3, qeye(N), qeye(2))
    a = tensor(qeye(3), destroy(N), qeye(2))
    x = tensor(qeye(3), create(N) + destroy(N), qeye(2)) / 2
    p = tensor(qeye(3), 1j * (create(N) - destroy(N)), qeye(2)) / 2

    rabi_rate = args['rabi_rate']
    rabi_rate_molecule = args['rabi_rate_molecule']
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol

    return (x * rabi_rate * np.sin(d * t) + p * rabi_rate * np.cos(d * t) +
            rabi_rate_molecule / 2 * (sigma_plus_mol * a.dag() * np.exp(-1j * d_mol * t) +
                                      sigma_minus_mol * a * np.exp(1j * d_mol * t)))


def H_mol3_atom_rsb(t, args):
    """
    Red sideband (RSB) Hamiltonian for the 3-level molecule + ion system.

    t : float
    args : dict
        N : int, motional Hilbert space dimension
        rabi_rate_atom : float, atomic Rabi frequency

    Returns
    -------
    Qobj : RSB Hamiltonian
    """
    N = args['N']
    sigma_plus_atom = tensor(qeye(3), qeye(N), sigmap())
    sigma_minus_atom = tensor(qeye(3), qeye(N), sigmam())
    a = tensor(qeye(3), destroy(N), qeye(2))

    rabi_rate_atom = args['rabi_rate_atom']
    return rabi_rate_atom / 2 * (sigma_minus_atom * a.dag() + sigma_plus_atom * a)


###################################################################################
########################## 2-LEVEL MOLECULE + ION #################################
###################################################################################

def H_mol_atom_odf(t, args):
    """
    ODF Hamiltonian for a 2-level molecule co-trapped with an ion.

    t : float
    args : dict
        N : int, motional Hilbert space dimension
        rabi_rate : float
        d : float, detuning

    Returns
    -------
    Qobj : ODF Hamiltonian
    """
    N = args['N']
    x = tensor(qeye(2), create(N) + destroy(N), qeye(2)) / 2
    p = tensor(qeye(2), 1j * (create(N) - destroy(N)), qeye(2)) / 2

    rabi_rate = args['rabi_rate']
    d = args['d']

    return x * rabi_rate * np.sin(d * t) + p * rabi_rate * np.cos(d * t)


def H_mol_atom_bsb(t, args):
    """
    BSB Hamiltonian for a 2-level molecule co-trapped with an ion.

    t : float
    args : dict
        N : int
        rabi_rate_molecule : float
        d : float
        w_mol : float

    Returns
    -------
    Qobj : BSB Hamiltonian
    """
    N = args['N']
    sigma_plus_mol = tensor(sigmap(), qeye(N), qeye(2))
    sigma_minus_mol = tensor(sigmam(), qeye(N), qeye(2))
    a = tensor(qeye(2), destroy(N), qeye(2))

    rabi_rate_molecule = args['rabi_rate_molecule']
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol

    return rabi_rate_molecule / 2 * (sigma_plus_mol * a.dag() * np.exp(-1j * d_mol * t) +
                                     sigma_minus_mol * a * np.exp(1j * d_mol * t))


def H_mol_atom_tot(t, args):
    """
    Combined BSB + ODF Hamiltonian for 2-level molecule + ion.

    t : float
    args : dict
        N : int
        rabi_rate : float
        rabi_rate_molecule : float
        d : float
        w_mol : float

    Returns
    -------
    Qobj : Total Hamiltonian
    """
    N = args['N']
    sigma_plus_mol = tensor(sigmap(), qeye(N), qeye(2))
    sigma_minus_mol = tensor(sigmam(), qeye(N), qeye(2))
    a = tensor(qeye(2), destroy(N), qeye(2))
    x = tensor(qeye(2), create(N) + destroy(N), qeye(2)) / 2
    p = tensor(qeye(2), 1j * (create(N) - destroy(N)), qeye(2)) / 2

    rabi_rate = args['rabi_rate']
    rabi_rate_molecule = args['rabi_rate_molecule']
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol

    return (x * rabi_rate * np.sin(d * t) + p * rabi_rate * np.cos(d * t) +
            rabi_rate_molecule / 2 * (sigma_plus_mol * a.dag() * np.exp(-1j * d_mol * t) +
                                      sigma_minus_mol * a * np.exp(1j * d_mol * t)))


def H_mol_atom_rsb(t, args):
    """
    RSB Hamiltonian for 2-level molecule + ion.

    t : float
    args : dict
        N : int
        rabi_rate_atom : float

    Returns
    -------
    Qobj : RSB Hamiltonian
    """
    N = args['N']
    sigma_plus_atom = tensor(qeye(2), qeye(N), sigmap())
    sigma_minus_atom = tensor(qeye(2), qeye(N), sigmam())
    a = tensor(qeye(2), destroy(N), qeye(2))

    rabi_rate_atom = args['rabi_rate_atom']
    return rabi_rate_atom / 2 * (sigma_minus_atom * a.dag() + sigma_plus_atom * a)


###################################################################################
########################## MOLECULE + MOTION ONLY #################################
###################################################################################

def H_mol_odf(t, args):
    """
    ODF Hamiltonian for molecule alone.

    t : float
    args : dict
        N : int
        rabi_rate : float
        d : float

    Returns
    -------
    Qobj : ODF Hamiltonian
    """
    N = args['N']
    x = tensor(qeye(2), create(N) + destroy(N)) / 2
    p = tensor(qeye(2), 1j * (create(N) - destroy(N))) / 2
    rabi_rate = args['rabi_rate']
    d = args['d']

    return x * rabi_rate * np.sin(d * t) + p * rabi_rate * np.cos(d * t)


def H_mol_bsb(t, args):
    """
    BSB Hamiltonian for molecule alone.

    t : float
    args : dict
        N : int
        rabi_rate_molecule : float
        d : float
        w_mol : float

    Returns
    -------
    Qobj : BSB Hamiltonian
    """
    N = args['N']
    sigma_plus = tensor(sigmap(), qeye(N))
    sigma_minus = tensor(sigmam(), qeye(N))
    a = tensor(qeye(2), destroy(N))

    rabi_rate_molecule = args['rabi_rate_molecule']
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol

    return rabi_rate_molecule / 2 * (sigma_plus * a.dag() * np.exp(-1j * d_mol * t) +
                                     sigma_minus * a * np.exp(1j * d_mol * t))


def H_mol_tot(t, args):
    """
    Combined BSB + ODF Hamiltonian for molecule alone.

    t : float
    args : dict
        N : int
        rabi_rate : float
        rabi_rate_molecule : float
        d : float
        w_mol : float

    Returns
    -------
    Qobj : Total Hamiltonian
    """
    N = args['N']
    x = tensor(qeye(2), create(N) + destroy(N)) / 2
    p = tensor(qeye(2), 1j * (create(N) - destroy(N))) / 2
    sigma_plus = tensor(sigmap(), qeye(N))
    sigma_minus = tensor(sigmam(), qeye(N))
    a = tensor(qeye(2), destroy(N))

    rabi_rate = args['rabi_rate']
    rabi_rate_molecule = args['rabi_rate_molecule']
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol

    return (x * rabi_rate * np.sin(d * t) + p * rabi_rate * np.cos(d * t) +
            rabi_rate_molecule / 2 * (sigma_plus * a.dag() * np.exp(-1j * d_mol * t) +
                                      sigma_minus * a * np.exp(1j * d_mol * t)))