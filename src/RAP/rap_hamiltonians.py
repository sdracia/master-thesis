# File: rap_hamiltonians.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Hamiltonian Module for Rapid Adiabatic Passage (RAP) Simulations.

This module provides time-dependent Hamiltonian constructions for a 2-level 
molecular system coupled to a motional mode. It includes models for:
- Blue Sideband (BSB) transitions with Gaussian envelopes and frequency chirps.
- Carrier transitions (no motional coupling).
- Standard Rabi flopping (constant detuning).
"""

import numpy as np
from qutip import *
from typing import Any, Dict
from RAP.rap_operators import RAP_sigmap_2


def RAP_bsb_H_2levels(t: float, args: Dict[str, Any]) -> Qobj:
    """
    Time-dependent Hamiltonian for a 2-level molecule interacting with a BSB laser.

    The Hamiltonian accounts for a Blue Sideband transition where the pulse 
    features a Gaussian intensity envelope and a linear frequency chirp (RAP). 
    It can optionally include an off-resonant carrier contribution.

    Parameters
    ----------
    t : float
        Current time step in the simulation.
    args : dict
        Dictionary of physical parameters:
        - 'n_motional' (int): Dimension of the motional Fock space.
        - 'coupling' (float): Transition coupling constant.
        - 'w_mol' (float): Molecular transition frequency.
        - 'D' (float): Total frequency sweep range (chirp).
        - 'T' (float): Interaction time (pulse center is at T/2).
        - 'sigma' (float): Standard deviation of the Gaussian pulse envelope.
        - 'rabi_rate' (float): Peak Rabi rate.
        - 'laser_detuning' (float): Fixed laser detuning offset.
        - 'off_resonant' (bool): Whether to include carrier interference.
        - 'lamb_dicke' (float): Lamb-Dicke parameter (if off_resonant is True).
        - 'trap_freq' (float): Motional trap frequency.

    Returns
    -------
    Qobj
        The Hamiltonian operator at time t.

    Raises
    ------
    ValueError
        If off_resonant is True but lamb_dicke is not provided.
    """
    # Extract structural and physical parameters
    n_motional = args['n_motional']
    j = args['j']
    coupling = args['coupling']
    w_mol = args['w_mol']
    final_time = args['final_time']
    D = args['D']
    sigma = args['sigma']
    T = args['T']
    lamb_dicke = args.get('lamb_dicke')
    off_resonant = args.get('off_resonant', False)
    w_trap = args.get('trap_freq')

    # Define Hilbert space operators
    sigma_plus_mol = tensor(RAP_sigmap_2(), qeye(n_motional))
    a = tensor(qeye(2), destroy(n_motional))

    # Time-dependent Rabi rate with Gaussian envelope
    rr = args['rabi_rate']
    rr = rr * np.exp(-(t - T/2)**2 / (2 * sigma**2))
    
    # Calculate detuning and time-dependent sweep (chirp)
    laser_detuning = args['laser_detuning']
    det_bsb = laser_detuning - w_mol
    delta_t_bsb = (D / T) * (t - T / 2) + det_bsb

    # Blue Sideband Hamiltonian term: sigma+ * a_dagger
    H_term_bsb = sigma_plus_mol * a.dag() * np.exp(-1j * delta_t_bsb * t)
    H_bsb = rr * np.abs(coupling) / 2 * (H_term_bsb + H_term_bsb.dag())

    # Include off-resonant carrier contribution if requested
    if off_resonant:
        if lamb_dicke is None:
            raise ValueError("lamb_dicke must be provided when off_resonant is True")
            
        # Carrier detuning relative to the motional sideband
        det_carrier = laser_detuning - (-w_trap + w_mol)
        delta_t_carrier = (D / T) * (t - T / 2) + det_carrier

        # Carrier term is suppressed by the Lamb-Dicke factor (1/eta)
        H_term_carrier = sigma_plus_mol * np.exp(-1j * delta_t_carrier * t)
        H_carrier = (rr * np.abs(coupling) / 2 / lamb_dicke * 
                     (H_term_carrier + H_term_carrier.dag()))
    
        return H_bsb + H_carrier
    
    return H_bsb


def RAP_carrier_H_2levels(t: float, args: Dict[str, Any]) -> Qobj:
    """
    Time-dependent Hamiltonian for a 2-level molecule interacting with a carrier.

    This represents the RAP process strictly for the internal states without 
    motional coupling (no phonon creation or annihilation).

    Parameters
    ----------
    t : float
        Current time step.
    args : dict
        Dictionary of parameters (similar to RAP_bsb_H_2levels).

    Returns
    -------
    Qobj
        The carrier Hamiltonian operator at time t.
    """
    n_motional = args['n_motional']
    coupling = args['coupling']
    w_mol = args['w_mol']
    D = args['D']
    sigma = args['sigma']
    T = args['T']

    # Internal raising operator in the combined space
    sigma_plus_mol = tensor(RAP_sigmap_2(), qeye(n_motional))

    # Gaussian Rabi rate envelope
    rr = args['rabi_rate']
    rr = rr * np.exp(-(t - T/2)**2 / (2 * sigma**2))
    
    # Linear frequency sweep (RAP) logic
    laser_detuning = args['laser_detuning']
    det_carrier = laser_detuning - w_mol
    delta_t_carrier = (D / T) * (t - T / 2) + det_carrier

    # Carrier term logic (no 'a' or 'a.dag()' operators)
    H_term_carrier = sigma_plus_mol * np.exp(-1j * delta_t_carrier * t)
    H_carrier = rr * np.abs(coupling) / 2 * (H_term_carrier + H_term_carrier.dag())
    
    return H_carrier


def RAP_rabiflop_H_2levels(t: float, args: Dict[str, Any]) -> Qobj:
    """
    Hamiltonian for standard Rabi flopping (constant parameters).

    Unlike the RAP Hamiltonians, this function assumes a constant detuning 
    (no chirp) and a constant Rabi rate (no Gaussian envelope).

    Parameters
    ----------
    t : float
        Current time step.
    args : dict
        Dictionary of parameters including 'off_resonant' carrier options.

    Returns
    -------
    Qobj
        The Rabi flop Hamiltonian at time t.

    Raises
    ------
    ValueError
        If off_resonant is True but lamb_dicke is not provided.
    """
    n_motional = args['n_motional']
    coupling = args['coupling']
    w_mol = args['w_mol']
    D = args['D']
    sigma = args['sigma']
    T = args['T']
    lamb_dicke = args.get('lamb_dicke')
    off_resonant = args.get('off_resonant', False)
    w_trap = args.get('trap_freq')

    sigma_plus_mol = tensor(RAP_sigmap_2(), qeye(n_motional))
    a = tensor(qeye(2), destroy(n_motional))

    # Constant Rabi rate (no time-dependent envelope)
    rr = args['rabi_rate']
    laser_detuning = args['laser_detuning']
    det_bsb = laser_detuning - w_mol

    # Blue sideband term with constant detuning
    H_term_bsb = sigma_plus_mol * a.dag() * np.exp(-1j * det_bsb * t)
    H_bsb = rr * np.abs(coupling) / 2 * (H_term_bsb + H_term_bsb.dag())

    if off_resonant:
        if lamb_dicke is None:
            raise ValueError("lamb_dicke must be provided when off_resonant is True")
            
        # Carrier detuning with a standard sweep centered at T/2
        det_carrier = laser_detuning - (-w_trap + w_mol)
        delta_t_carrier = (D / T) * (t - T / 2) + det_carrier

        H_term_carrier = sigma_plus_mol * np.exp(-1j * delta_t_carrier * t)
        H_carrier = (rr * np.abs(coupling) / 2 / lamb_dicke * 
                     (H_term_carrier + H_term_carrier.dag()))
        
        return H_bsb + H_carrier
    
    return H_bsb