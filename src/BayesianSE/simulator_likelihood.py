# File: simulator_likelihood.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Likelihood and State Transition Module for the Bayesian Simulator.

This module provides the core physical simulation logic for:
- Calculating the excitation probability matrix for a single molecule pulse.
- Incorporating systematic laser miscalibration and shot-to-shot noise.
- Sampling the next physical state index from a probability distribution.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_array as sparray
from scipy.sparse import diags, csr_array
from typing import Dict, Optional, Any

from exp_imperfections import apply_noise
from ._utils import checks_likelihoods


def likelihoods_simulator(
    self,
    frequency: float,
    duration_us: float,
    rabi_rate_mhz: float,
    dephased: bool = False,
    coherence_time_us: float = 1000.0,
    is_minus: bool = True,
    noise_params: Optional[Dict[str, Dict[str, float]]] = None,
    seed: Optional[int] = None,
    maximum_excitation: float = 0.9,
    laser_miscalibration: Optional[Dict[str, Dict[str, float]]] = None,   
    seed_miscalibration: Optional[int] = None
) -> sparray:
    """
    Computes the transition matrix for the physical simulator.

    This function represents the 'real' physics of the system. It calculates 
    how a pulse redistributes populations across the molecular states, 
    accounting for experimental imperfections.

    Parameters
    ----------
    frequency : float
        Raman difference frequency of the pulse in MHz.
    duration_us : float
        Pulse duration in microseconds.
    rabi_rate_mhz : float
        Rabi rate in MHz.
    dephased : bool, optional
        Whether the excitation is modeled as dephased. Default is False.
    coherence_time_us : float, optional
        Coherence time for dephasing in microseconds. Default is 1000.
    is_minus : bool, optional
        If True, targets Delta_m = -1 transitions. Default is True.
    noise_params : dict, optional
        Parameters for shot-to-shot frequency and Rabi rate fluctuations.
    seed : int, optional
        Seed for shot-to-shot noise.
    maximum_excitation : float, optional
        Upper bound for transition probability. Default is 0.9.
    laser_miscalibration : dict, optional
        Parameters for systematic laser miscalibration.
    seed_miscalibration : int, optional
        Seed for miscalibration noise.

    Returns
    -------
    exc_matrix_checked : sparray
        Sparse matrix representing the transition probabilities between states.
    """
    original_freq = frequency   

    if laser_miscalibration is None:
        laser_miscalibration = {}

    if "frequency" in laser_miscalibration:
        frequency = apply_noise(
            frequency, 
            laser_miscalibration["frequency"]["type"], 
            laser_miscalibration["frequency"]["level"], 
            seed_miscalibration
        )
    if "rabi_rate" in laser_miscalibration:
        rabi_rate_mhz = apply_noise(
            rabi_rate_mhz, 
            laser_miscalibration["rabi_rate"]["type"], 
            laser_miscalibration["rabi_rate"]["level"], 
            seed_miscalibration
        )

    if noise_params is None:
        noise_params = {}

    if "frequency" in noise_params:
        frequency = apply_noise(
            frequency, 
            noise_params["frequency"]["type"], 
            noise_params["frequency"]["level"], 
            seed
        )
    if "rabi_rate" in noise_params:
        rabi_rate_mhz = apply_noise(
            rabi_rate_mhz, 
            noise_params["rabi_rate"]["type"], 
            noise_params["rabi_rate"]["level"], 
            seed
        )
    
    self.misfrequency = frequency - original_freq       

    num_states = len(self.model.state_df)

    if is_minus:
        detunings = 2 * np.pi * (
            frequency - self.model.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3
        )
    else:
        detunings = 2 * np.pi * (
            frequency + self.model.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3
        )

    omegas = rabi_rate_mhz * self.model.transition_df["coupling"].to_numpy(dtype=float)

    if dephased:
        transition_exc_probs = (
            maximum_excitation * omegas**2 / (omegas**2 + detunings**2) * 
            ((1 - np.cos(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us) * 
              np.exp(-duration_us / coherence_time_us)) / 2)
        )
    else:
        transition_exc_probs = (
            maximum_excitation * omegas**2 / (omegas**2 + detunings**2) * 
            np.sin(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us / 2) ** 2
        )

    if is_minus:
        rows = self.model.transition_df["index2"].to_numpy(dtype=int)
        cols = self.model.transition_df["index1"].to_numpy(dtype=int)
    else:
        rows = self.model.transition_df["index1"].to_numpy(dtype=int)
        cols = self.model.transition_df["index2"].to_numpy(dtype=int)
    
    exc_matrix = (
        diags([1.0] * num_states, offsets=0, format="csr") +
        csr_array((transition_exc_probs, (rows, cols)), shape=(num_states, num_states)) +
        csr_array((-transition_exc_probs, (cols, cols)), shape=(num_states, num_states))
    )

    exc_matrix_checked = checks_likelihoods(exc_matrix)

    return exc_matrix_checked