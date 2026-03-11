# File: simulator_dynamics.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Measurement Outcome Simulation Module for Bayesian Estimation.

This module provides the core logic for simulating the stochastic evolution 
of a single molecule subjected to a measurement pulse. It accounts for 
transition probabilities, state jumps, and experimental detection 
imperfections such as false positives and false negatives.
"""

import numpy as np
from typing import List, Dict, Optional, Any


def outcome_simulator(
    self, 
    current_measurement: List[Any], 
    noise_params: Optional[Dict[str, Dict[str, float]]] = None, 
    seed: Optional[int] = None, 
    laser_miscalibration: Optional[Dict[str, Dict[str, float]]] = None, 
    seed_miscalibration: Optional[int] = None,
    max_excitation: float = 0.9
) -> int:
    """
    Simulates the physical transition and measurement outcome of a molecule.

    The function determines if a measurement pulse induces a state jump 
    according to the calculated transition probabilities. It then simulates 
    the detection process by applying false positive and false negative rates 
    to determine the observed experimental outcome.

    Parameters
    ----------
    current_measurement : list
        A list containing pulse parameters: 
        [index, frequency, duration, dephased, coherence_time, is_minus, rabi_rate].
    noise_params : dict, optional
        Shot-to-shot noise parameters for frequency and Rabi rate.
    seed : int, optional
        Random seed for the shot-to-shot noise.
    laser_miscalibration : dict, optional
        Systematic laser calibration error parameters.
    seed_miscalibration : int, optional
        Random seed for the systematic miscalibration.
    max_excitation : float, optional
        Upper limit of the transition probability. Default is 0.9.

    Returns
    -------
    outcome : int
        The simulated experimental outcome (0 for dark/no-jump, 1 for bright/jump), 
        after applying experimental noise (FPR/FNR).
    """

    # --- 1. PARAMETER EXTRACTION ---
    frequency = current_measurement[1]
    duration = current_measurement[2]
    dephasing = current_measurement[3]
    coherence_time = current_measurement[4]
    is_minus = current_measurement[5]
    rabi_rate_mhz = current_measurement[6]

    current_state = self.history_list[-1]
    current_index = current_state[0]

    # --- 2. TRANSITION PROBABILITY CALCULATION ---
    exc_matrix = self.likelihoods_simulator(
        frequency=frequency,
        duration_us=duration,
        rabi_rate_mhz=rabi_rate_mhz,
        dephased=dephasing,
        coherence_time_us=coherence_time,
        is_minus=is_minus,
        noise_params=noise_params,
        seed=seed,
        maximum_excitation=max_excitation,
        laser_miscalibration=laser_miscalibration,
        seed_miscalibration=seed_miscalibration
    )

    # Extract the probability column corresponding to the current state.
    # This represents P(final_state | initial_state).
    column = exc_matrix[:, current_index]

    # --- 3. STOCHASTIC STATE EVOLUTION ---
    new_state_index = self.new_state_index(column)

    selected_row = self.model.state_df.loc[new_state_index]
    j_val = selected_row["j"]
    m_val = selected_row["m"]
    xi_val = selected_row["xi"]

    new_state = [new_state_index, j_val, m_val, xi_val]
    self.history_list.append(new_state)

    # --- 4. MEASUREMENT OUTCOME LOGIC ---
    if current_index == new_state_index:
        outcome = 0

        # Apply False Positive Rate (FPR): Dark state mistakenly detected as Bright
        if np.random.rand() < self.fpr:
            outcome = 1
    else:
        outcome = 1

        # Apply False Negative Rate (FNR): Bright state mistakenly detected as Dark
        if np.random.rand() < self.fnr:
            outcome = 0

    return outcome