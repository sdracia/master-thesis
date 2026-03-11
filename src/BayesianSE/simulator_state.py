# File: simulator_state.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Module for state initialization and stochastic sampling in molecular simulations.

This module provides functions to:
- Initialize the molecular state by sampling from thermal or pumped distributions.
- Sample new state indices based on transition probability vectors (sparse matrices).
"""

import numpy as np
from typing import Tuple, Any


def state_initialization(self, j_max: int) -> Tuple[int, int, float, bool]:
    """
    Initializes the molecular state by sampling from the global population distribution.

    This function first selects a rotational manifold (J) based on the thermal 
    or pumped distribution and then selects a specific state (m, xi) within that 
    manifold.

    Parameters
    ----------
    j_max : int
        The maximum rotational quantum number J to consider.

    Returns
    -------
    random_index : int
        The row index of the selected state in the molecular DataFrame.
    j_val : int
        The rotational quantum number J of the selected state.
    m_val : float
        The magnetic quantum number m of the selected state.
    xi_val : bool
        The parity/symmetry label (xi) of the selected state.
    """
    j_values = np.arange(j_max + 1)
    j_probs = self.j_distribution(self.model, j_max)

    assert np.isclose(np.sum(j_probs), 1.0), "Distribution is not normalized."

    j_sample = np.random.choice(j_values, p=j_probs)

    states_in_j = self.model.state_df.loc[
        self.model.state_df["j"] == j_sample
    ].copy()

    indices = states_in_j.index.values

    probabilities = states_in_j["state_dist"].values / j_probs[j_sample]

    total = probabilities.sum()
    assert np.isclose(total, 1.0), "Distribution inside manifold is not normalized."

    random_index = np.random.choice(indices, p=probabilities)

    selected_row = states_in_j.loc[random_index]
    j_val = selected_row["j"]
    m_val = selected_row["m"]
    xi_val = selected_row["xi"]

    print(
        f"Extracted level is idx={random_index}: J={j_val}, m={m_val}, xi={xi_val}", 
        '\n'
    )

    return random_index, j_val, m_val, xi_val


def new_state_index(
    self, 
    col_sparse: Any, 
    rng: np.random.Generator = np.random.default_rng()
) -> int:
    """
    Samples a row index from a sparse column vector representing a probability distribution.

    Used by the simulator to determine the next physical state after a 
    transition pulse.

    Parameters
    ----------
    col_sparse : scipy.sparse.spmatrix
        A sparse column vector (CSC or CSR format) of shape (N, 1).
    rng : numpy.random.Generator, optional
        Random number generator instance. Default is np.random.default_rng().

    Returns
    -------
    chosen_idx : int
        The sampled row index based on the weights in the sparse vector.
    """
    row_indices = col_sparse.nonzero()[0]
    values = col_sparse.data

    prob_dist = values / values.sum()

    chosen_idx = rng.choice(row_indices, p=prob_dist)
    
    return chosen_idx