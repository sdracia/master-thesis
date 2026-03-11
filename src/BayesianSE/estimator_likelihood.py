"""
Likelihood Calculation Module for Bayesian State Estimation.

This module provides the core logic to calculate likelihood matrices (L0 and L1)
used in Bayesian state estimation. These matrices represent the probabilities 
of observing a specific experimental outcome (e.g., bright or dark state) 
given the current hypothesis of the molecular state distribution.
"""

import numpy as np
from scipy.sparse import diags, csr_array
from scipy.sparse import csr_matrix as sparray
from typing import Dict, Tuple, Optional

from exp_imperfections import apply_noise
from ._utils import checks_likelihoods


def likelihoods_estimator(
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
) -> Tuple[sparray, sparray]:
    """
    Computes the likelihood matrices for a given experimental measurement setting.

    The likelihood matrices L0 (diagonal) and L1 (off-diagonal) are derived from 
    the transition excitation probabilities. L0 represents the probability 
    of remaining in the same state, while L1 represents the probability of 
    transitioning between states.

    Parameters
    ----------
    self : BayesianStateEstimation
        Instance of the Bayesian estimator containing the molecular model.
    frequency : float
        The probe laser frequency in MHz.
    duration_us : float
        The pulse duration in microseconds.
    rabi_rate_mhz : float
        The Rabi frequency in MHz.
    dephased : bool, optional
        Whether to use a dephased excitation model. Default is False.
    coherence_time_us : float, optional
        The coherence time for dephasing in microseconds. Default is 1000.
    is_minus : bool, optional
        Transition direction (Delta_m = -1 if True, Delta_m = +1 if False). 
        Default is True.
    noise_params : dict, optional
        Shot-to-shot noise parameters for frequency and Rabi rate.
    seed : int, optional
        Random seed for shot-to-shot noise.
    maximum_excitation : float, optional
        Upper limit for the transition probability (e.g., 0.9 for 90%). 
        Default is 0.9.
    laser_miscalibration : dict, optional
        Fixed miscalibration parameters for the experimental setup.
    seed_miscalibration : int, optional
        Random seed for the miscalibration noise.

    Returns
    -------
    likelihood0 : sparray
        Diagonal sparse matrix representing P(0|state).
    likelihood1 : sparray
        Off-diagonal sparse matrix representing P(1|state).
    """

    # --- 1. HANDLE SYSTEMATIC LASER MISCALIBRATION ---
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

    # --- 2. HANDLE SHOT-TO-SHOT FLUCTUATIONS ---
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

    # --- 3. COMPUTE TRANSITION PROBABILITIES ---
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
    
    # --- 4. CONSTRUCT THE EXCITATION MATRIX ---
    exc_matrix = (
        diags([1.0] * num_states, offsets=0, format="csr") +
        csr_array((transition_exc_probs, (rows, cols)), shape=(num_states, num_states)) +
        csr_array((-transition_exc_probs, (cols, cols)), shape=(num_states, num_states))
    )

    exc_matrix_checked = checks_likelihoods(exc_matrix)

    # --- 5. SPLIT INTO BAYESIAN LIKELIHOODS ---
    diagonal_matrix = diags(exc_matrix_checked.diagonal(), format='csr')
    off_diagonal_matrix = exc_matrix_checked - diagonal_matrix

    likelihood0 = diagonal_matrix
    likelihood1 = off_diagonal_matrix

    return likelihood0, likelihood1