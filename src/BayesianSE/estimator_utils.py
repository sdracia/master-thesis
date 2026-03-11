# File: estimator_utils.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Utility module for Bayesian State Estimation metrics and control.

This module provides helper functions for:
- Initializing the Bayesian prior based on the current model distribution.
- Defining convergence criteria for the iterative estimation process.
- Calculating the Cross Entropy as a performance metric to evaluate the 
  accuracy of the estimator against the ground-truth simulator.
"""

import numpy as np


def init_prior(self):
    """
    Initializes the estimator's prior distribution.

    Sets the starting belief of the estimator by copying the 'state_dist' 
    column from the molecular model's dataframe.

    Parameters
    ----------
    self : BayesianStateEstimation
        The instance of the Bayesian estimator.
    """
    self.prior = self.model.state_df["state_dist"]


def stop_condition(self, posterior: np.ndarray) -> bool:
    """
    Evaluates if the estimation process has converged.

    The condition is satisfied if any single state in the posterior 
    distribution has a probability exceeding 90%.

    Parameters
    ----------
    posterior : np.ndarray
        The current posterior probability distribution.

    Returns
    -------
    bool
        True if the convergence threshold is reached, False otherwise.
    """
    return any(p > 0.9 for p in posterior)


def cross_entropy(self, simulator: any) -> list:
    """
    Calculates the Cross Entropy between the estimator's belief and reality.

    This metric measures how well the estimated posterior distribution 
    predicts the true physical state occupied by the molecule in the simulator.

    Parameters
    ----------
    simulator : any
        The physical simulator object containing the 'true' history of states.

    Returns
    -------
    cross_entropies : list
        A list of cross entropy values calculated for each step of the history.

    Raises
    ------
    ValueError
        If a negative Cross Entropy value is encountered during calculation.
    """
    posteriors = [np.array(entry["posterior"]) for entry in self.history_list]

    cross_entropies = []
    
    for i in range(len(simulator.history_list[1:])):
        index = simulator.history_list[i + 1][0]
        posterior = posteriors[i]
        
        ce_value = -np.log(posterior[index] + 1e-11)  

        if ce_value < 0 and np.isclose(ce_value, 0, atol=1e-10):
            ce_value = 1e-11

        cross_entropies.append(ce_value)

    for i, ce in enumerate(cross_entropies):
        if ce < 0:
            raise ValueError(f"Negative Cross Entropy found at index {i}: {ce}")

    return cross_entropies