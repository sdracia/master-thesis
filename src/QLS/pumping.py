# File: pumping.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Module for pumping and excitation of molecular states.

This module provides functions to:
- Build excitation matrices for molecular transitions.
- Apply repeated pumping sequences to redistribute populations among states.
- Include experimental imperfections such as noise and laser miscalibration.
"""

import numpy as np
from molecules.molecule import Molecule
from scipy.sparse import csr_array, sparray
from typing import Optional, Dict
import exp_imperfections as imp


def excitation_matrix(
    molecule: Molecule,
    frequency: float,
    duration_us: float,
    rabi_rate_mhz: float,
    dephased: bool = False,
    coherence_time_us: float = 1000.0,
    is_minus: bool = True,
    noise_params: Optional[Dict[str, Dict[str, float]]] = None,
    seed: Optional[int] = None,
    laser_miscalibration: Optional[Dict[str, Dict[str, float]]] = None,   
    seed_miscalibration: Optional[int] = None
) -> sparray:
    """
    Constructs the excitation probability matrix for a single pulse.

    Parameters
    ----------
    molecule : Molecule
        The molecule object containing state and transition data.
    frequency : float
        Raman difference frequency of the excitation pulse in MHz.
    duration_us : float
        Duration of the excitation pulse in microseconds.
    rabi_rate_mhz : float
        Rabi rate in MHz.
    dephased : bool, optional
        Whether to use dephased excitation formula. Default is False.
    coherence_time_us : float, optional
        Coherence time for dephasing in microseconds. Default is 1000.
    is_minus : bool, optional
        If True, calculates Δm = -1 transitions. Default is True.
    noise_params : dict, optional
        Dictionary specifying noise parameters for frequency and Rabi rate.
    seed : int, optional
        Seed for random noise.
    laser_miscalibration : dict, optional
        Dictionary specifying laser miscalibration parameters.
    seed_miscalibration : int, optional
        Seed for miscalibration noise.

    Returns
    -------
    exc_matrix : sparray
        Sparse matrix representing the excitation probabilities between states.
    """
    if laser_miscalibration is None:
        laser_miscalibration = {}

    if "frequency" in laser_miscalibration:
        frequency = imp.apply_noise(
            frequency,
            laser_miscalibration["frequency"]["type"],
            laser_miscalibration["frequency"]["level"],
            seed_miscalibration
        )
    if "rabi_rate" in laser_miscalibration:
        rabi_rate_mhz = imp.apply_noise(
            rabi_rate_mhz,
            laser_miscalibration["rabi_rate"]["type"],
            laser_miscalibration["rabi_rate"]["level"],
            seed_miscalibration
        )

    if noise_params is None:
        noise_params = {}

    # Apply noise to frequency and Rabi rate if specified
    if "frequency" in noise_params:
        frequency = imp.apply_noise(
            frequency,
            noise_params["frequency"]["type"],
            noise_params["frequency"]["level"],
            seed
        )
    if "rabi_rate" in noise_params:
        rabi_rate_mhz = imp.apply_noise(
            rabi_rate_mhz,
            noise_params["rabi_rate"]["type"],
            noise_params["rabi_rate"]["level"],
            seed
        )

    num_states = len(molecule.state_df)

    # Compute detunings based on transition direction
    if is_minus:
        detunings = 2 * np.pi * (frequency - molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)
    else:
        detunings = 2 * np.pi * (frequency + molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)

    omegas = rabi_rate_mhz * molecule.transition_df["coupling"].to_numpy(dtype=float)

    # Compute excitation probabilities per transition
    if dephased:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * (
            (1 - np.cos(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us) *
             np.exp(-duration_us / coherence_time_us)) / 2
        )
    else:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * \
            np.sin(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us / 2) ** 2

    # Construct sparse excitation matrix
    if is_minus:
        rows = molecule.transition_df["index2"].to_numpy(dtype=int)
        cols = molecule.transition_df["index1"].to_numpy(dtype=int)
        exc_matrix = csr_array(
            (transition_exc_probs, (rows, cols)), shape=(num_states, num_states)
        ) + csr_array(
            (-transition_exc_probs, (cols, cols)), shape=(num_states, num_states)
        )
    else:
        rows = molecule.transition_df["index1"].to_numpy(dtype=int)
        cols = molecule.transition_df["index2"].to_numpy(dtype=int)
        exc_matrix = csr_array(
            (transition_exc_probs, (rows, cols)), shape=(num_states, num_states)
        ) + csr_array(
            (-transition_exc_probs, (cols, cols)), shape=(num_states, num_states)
        )

    return exc_matrix


def apply_pumping(
    molecule: Molecule,
    pump_frequency_mhz: float,
    num_pumps: int,
    pump_duration_us: float,
    pump_rabi_rate_mhz: float,
    pump_dephased: bool = False,
    coherence_time_us: float = 1000.0,
    is_minus: bool = True,
    noise_params: Optional[Dict[str, Dict[str, float]]] = None,
    seed: Optional[int] = None,
    laser_miscalibration: Optional[Dict[str, Dict[str, float]]] = None,   
    seed_miscalibration: Optional[int] = None
) -> None:
    """
    Applies repeated pumping pulses to update the molecular state distribution.

    Parameters
    ----------
    molecule : Molecule
        Molecule object containing state distribution.
    pump_frequency_mhz : float
        Raman difference frequency of the pumping pulse in MHz.
    num_pumps : int
        Number of repeated pumping pulses.
    pump_duration_us : float
        Duration of each pulse in microseconds.
    pump_rabi_rate_mhz : float
        Rabi rate of the pump in MHz.
    pump_dephased : bool, optional
        Whether pulses are dephased. Default is False.
    coherence_time_us : float, optional
        Coherence time for dephased pulses. Default is 1000.
    is_minus : bool, optional
        If True, applies Δm = -1 transitions. Default is True.
    noise_params : dict, optional
        Noise parameters for frequency and Rabi rate.
    seed : int, optional
        Seed for noise.
    laser_miscalibration : dict, optional
        Laser miscalibration parameters.
    seed_miscalibration : int, optional
        Seed for miscalibration noise.

    Raises
    ------
    ValueError
        If excitation matrix sum is non-zero, negative populations appear,
        or shape mismatch occurs between state distribution and excitation matrix.
    """
    for _ in range(num_pumps):
        exc_matrix = excitation_matrix(
            molecule, pump_frequency_mhz, pump_duration_us, pump_rabi_rate_mhz,
            pump_dephased, coherence_time_us, is_minus, noise_params, seed,
            laser_miscalibration, seed_miscalibration
        ).dot(molecule.state_df["state_dist"])
        molecule.state_df["state_dist"] += exc_matrix

        mask = molecule.state_df["state_dist"] < 0
        if np.abs(sum(exc_matrix)) >= 1e-10:
            raise ValueError("Error: sum of exc_matrix is not 0")

        if (molecule.state_df["state_dist"].shape) != (exc_matrix.shape):
            raise ValueError(f"Error: Shape mismatch. state_dist has shape {molecule.state_df['state_dist'].shape}, but exc_matrix has shape {exc_matrix.shape}")

        if np.sum(mask) > 0: 
            raise ValueError("Error: state_dist contains negative values")