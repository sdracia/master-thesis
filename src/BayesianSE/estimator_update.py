# File: estimator_update.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Module for Bayesian Probability Distribution Updates in Molecular State Estimation.

This module contains the core logic for iteratively updating the prior distribution 
based on simulated measurement outcomes. It handles measurement block scheduling, 
incorporates experimental imperfections (FPR/FNR), and manages re-pumping conditions 
to maintain the validity of the estimation over long sequences.
"""

import numpy as np
from molecules.molecule import CaH, CaOH
from typing import Any
from ._statistics import compute_variance


def update_distibution(
    self, 
    simulator: Any, 
    num_updates: int, 
    apply_pumping: bool = False, 
    save_data: bool = True, 
    block_steps: int = 1, 
    type_block: str = None, 
    noise_params: dict = None, 
    seed: int = None, 
    laser_miscalibration: dict = None, 
    seed_miscalibration: int = None,
    pop_fit: np.ndarray = None,
    false_rates: bool = True,
    max_excitation: float = 0.9
):
    """
    Updates the Bayesian prior distribution based on measurement outcomes.

    This function iterates through a sequence of measurement blocks, calls the 
    simulator to obtain experimental outcomes, and applies the Bayesian update 
    rule to calculate the posterior distribution.

    Parameters
    ----------
    simulator : Any
        The physical simulator or experimental interface providing outcomes.
    num_updates : int
        Number of global update cycles (sweeps).
    apply_pumping : bool, optional
        If True, applies repumping logic during the sequence. Default is False.
    save_data : bool, optional
        Whether to record the history of updates. Default is True.
    block_steps : int, optional
        Number of measurements within a single block. Default is 1.
    type_block : str, optional
        The specific measurement pattern (e.g., 'block1', 'block2').
    noise_params : dict, optional
        Shot-to-shot noise parameters for the simulator.
    seed : int, optional
        Random seed for the simulator noise.
    laser_miscalibration : dict, optional
        Fixed laser miscalibration parameters.
    seed_miscalibration : int, optional
        Random seed for miscalibration effects.
    pop_fit : np.ndarray, optional
        Efficiencies used for the pumping model.
    false_rates : bool, optional
        Whether to incorporate False Positive and False Negative rates in 
        the likelihood. Default is True.
    max_excitation : float, optional
        Maximum possible excitation probability for the pulse. Default is 0.9.

    Raises
    ------
    ValueError
        If apply_pumping is True but pop_fit is not provided.
    """
    
    # Species-specific variance thresholds to trigger re-pumping
    if isinstance(self.model, CaH):
        threshold_variance = 2500
    elif isinstance(self.model, CaOH):
        threshold_variance = 0.15 * 1e6

    if apply_pumping:
        if pop_fit is not None: 
            self.pump_efficiencies = pop_fit
        else:
            raise ValueError("Need efficiencies for pumping")

    final_step = num_updates * block_steps
    stop_flag = False
    variance = 0.0

    for j in range(num_updates):

        # Logic for 'within-run' pumping: if the estimator's belief is too spread 
        # (high variance) after a full sweep, we reset the physical state and belief.
        if apply_pumping:
            if j != 0 and j % self.j_max == 0 and variance > threshold_variance:
                print("Re-pumping updating prior")
                self.within_run_pumping(simulator, save_data)
                
        for i in range(block_steps):
            
            # Determine which measurement setting to use
            self.meas_idx = self.get_next_setting(j, i, ty=type_block)

            data = self.prior
            lh0 = self.Probs_exc_list[self.meas_idx][0]
            lh1 = self.Probs_exc_list[self.meas_idx][1]

            current_measurement = self.measurements[self.meas_idx]

            # Obtain physical outcome from the simulator (0 or 1)
            self.outcome = simulator.outcome_simulator(
                current_measurement, 
                noise_params, 
                seed, 
                laser_miscalibration, 
                seed_miscalibration,
                max_excitation
            )

            # Bayesian Likelihood adjustment for experimental imperfections (FPR/FNR)
            if false_rates:
                if self.outcome == 0:
                    likelihood = (1 - self.fnr) * lh0 + self.fpr * lh1
                else:
                    likelihood = self.fnr * lh0 + (1 - self.fpr) * lh1
            else:
                if self.outcome == 0:
                    likelihood = lh0
                else:
                    likelihood = lh1
            
            # Apply Bayesian update: Posterior proportional to Likelihood * Prior
            posterior = likelihood.dot(data)
            posterior = posterior / np.sum(posterior)

            self.likelihood = likelihood
            self.posterior = posterior

            if save_data:
                self.history_list.append({
                    "meas_idx": self.meas_idx,
                    "measurement": self.measurements[self.meas_idx],
                    "outcome": self.outcome,
                    "prior": self.prior.tolist(),
                    "likelihood": self.likelihood,
                    "posterior": self.posterior.tolist()
                })

            # The current posterior becomes the prior for the next iteration
            self.prior = posterior
            variance = compute_variance(self.prior)

            # Check if the estimation has converged
            if self.stop_condition(self.prior):
                final_step = (j + 1) * block_steps
                print(f"Stop condition reached at step {final_step}")
                stop_flag = True
                break

        if stop_flag:
            break


def get_next_setting(self, j: int, i: int, ty: str = None) -> int:
    """
    Selects the measurement index based on the current step and block type.

    Parameters
    ----------
    j : int
        The current global update cycle index.
    i : int
        The current step index within the measurement block.
    ty : str, optional
        The type of block pattern to use (e.g., 'block1', 'block5').

    Returns
    -------
    idx : int
        The index of the measurement setting to be used.
    """
    j_max = self.j_max

    # Logic for direct vs inverse measurement selection (cycling through manifolds)
    direct_meas = j_max - 1 - (j % j_max)
    inverse_meas = j_max - 1 - (j % j_max) + j_max

    if ty is None:
        idx = direct_meas

    # Pattern: u-d-d
    if ty == "block1":
        if i == 0:
            idx = direct_meas
        else:
            idx = inverse_meas

    # Pattern: u-d-u
    if ty == "block2":
        if i == 0 or i == 2:    
            idx = direct_meas
        else:
            idx = inverse_meas

    # Pattern: u-d
    if ty == "block3":
        if i == 0:
            idx = direct_meas
        else:
            idx = inverse_meas

    # Pattern: u-d-u-d
    if ty == "block4":
        if i == 0 or i == 2:
            idx = direct_meas
        else:
            idx = inverse_meas

    # Pattern: u-d-d-u-d-d
    if ty == "block5":
        if i == 0 or i == 3:
            idx = direct_meas
        else:
            idx = inverse_meas

    # Pattern: u-u-u-d-d-d-u-u-u
    if ty == "block6":
        if i == 0 or i == 1 or i == 2 or i == 6 or i == 7 or i == 8:   
            idx = direct_meas
        else:
            idx = inverse_meas

    # Pattern: u-u-u-u-d-d-d-d-u-u-u-u
    if ty == "block7":
        if i == 0 or i == 1 or i == 2 or i == 3 or i == 8 or i == 9 or i == 10 or i == 11:
            idx = direct_meas
        else:
            idx = inverse_meas

    # Pattern: u-d-u-d-u-d-u-d-u-d-u-d-u-d-u-d-u-d-u-d-u-d-u-d
    if ty == "block8":
        if i == 0 or i == 2 or i == 4 or i == 6 or i == 8 or i == 10 or i == 12 or i == 14 or i == 16 or i == 18 or i == 20 or i == 22:
            idx = direct_meas
        else:
            idx = inverse_meas

    # Pattern: u-u-u-u-d-d-d-d-u-u-u-u-d-d-d-d-u-u-u-u-d-d-d-d
    if ty == "block9":
        if i == 0 or i == 1 or i == 2 or i == 3 or i == 8 or i == 9 or i == 10 or i == 11 or i == 16 or i == 17 or i == 18 or i == 19:
            idx = direct_meas
        else:
            idx = inverse_meas

    return idx
