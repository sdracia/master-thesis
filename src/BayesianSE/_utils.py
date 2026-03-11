# File: _utils.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Utility module for matrix validation and convergence analysis.

This module provides functions to:
- Verify the physical consistency of transition probability matrices (likelihoods).
- Process and filter simulation results based on convergence thresholds to 
  evaluate estimator performance.
"""

import numpy as np
from scipy.sparse import csr_array, diags
from typing import Dict, List, Tuple, Any


def checks_likelihoods(
    exc_matrix: csr_array
) -> csr_array:
    """
    Validates and renormalizes a transition probability matrix.

    This function ensures that the matrix contains no significant negative 
    values and that each column (state-out probability) sums to approximately 1, 
    preserving physical probability conservation.

    Parameters
    ----------
    exc_matrix : csr_array
        The sparse transition matrix to be checked.

    Returns
    -------
    exc_matrix : csr_array
        The validated and potentially renormalized sparse matrix.

    Raises
    ------
    ValueError
        If significant negative values are found or if the sum of probabilities 
        deviates significantly from unity.
    """
    # === 1. Check for negative values ===
    tol_neg = 1e-6  
    negative_mask = exc_matrix.data < -tol_neg

    if np.any(negative_mask):
        raise ValueError(
            f"Significant negative values found in the matrix: "
            f"{exc_matrix.data[negative_mask]}"
        )
    else:
        small_neg_mask = (exc_matrix.data < 0) & (exc_matrix.data >= -tol_neg)
        exc_matrix.data[small_neg_mask] = 0.0

    # === 2. Check for probability conservation (sum ≈ 1) ===
    row_sums = np.array(exc_matrix.sum(axis=0)).flatten()
    tol_row = 1e-3  
    bad_rows = np.abs(row_sums - 1.0) > tol_row

    if np.any(bad_rows):
        raise ValueError(
            f"The sum of some rows is significantly different from 1: "
            f"{row_sums[bad_rows]}"
        )
    else:
        # Perform fine-grained renormalization if deviation is small but present
        needs_norm = (np.abs(row_sums - 1.0) > 1e-12)
        if np.any(needs_norm):
            correction_factors = np.ones_like(row_sums)
            correction_factors[needs_norm] = 1.0 / row_sums[needs_norm]
            
            # Apply correction via diagonal scaling matrix
            row_scaling = diags(correction_factors)
            exc_matrix = row_scaling @ exc_matrix 

    return exc_matrix


def cleaning_convergence(
    max_step: int, 
    curves_by_label: Dict[str, List[Tuple[np.ndarray, np.ndarray]]], 
    misfrequency_by_label: Dict[str, List[Any]], 
    variance_by_label: Dict[str, List[Any]]
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, float, float]:
    """
    Filters simulation runs based on their convergence behavior.

    A run is considered 'converged' if it reached the target criteria before 
    a threshold defined as 97% of the maximum allowed steps.

    Parameters
    ----------
    max_step : int
        The maximum number of steps allowed in the simulation.
    curves_by_label : dict
        Mapping of labels to lists of (steps, cross-entropy) curves.
    misfrequency_by_label : dict
        Mapping of labels to recorded frequency errors.
    variance_by_label : dict
        Mapping of labels to recorded estimator variances.

    Returns
    -------
    tuple
        A long tuple containing filtered dictionaries for converged runs, 
        dictionaries for non-converged runs, and the calculated 
        fractions of convergence.
    """
    # Threshold defined to distinguish between successful convergence and timeout
    threshold = max_step - int(0.03 * max_step)

    # Initialize containers for filtered data
    curves_by_label_filtered = {}
    misfrequency_by_label_filtered = {}
    variance_by_label_filtered = {}

    curves_by_label_not_converged = {}
    misfrequency_by_label_not_converged = {}
    variance_by_label_not_converged = {}

    total_runs = 0
    converged_runs = 0

    for label in curves_by_label:
        curves_by_label_filtered[label] = []
        misfrequency_by_label_filtered[label] = []
        variance_by_label_filtered[label] = []

        curves_by_label_not_converged[label] = []
        misfrequency_by_label_not_converged[label] = []
        variance_by_label_not_converged[label] = []

        for i, (steps, ce_curve) in enumerate(curves_by_label[label]):
            last_step = steps[-1] if len(steps) > 0 else -1
            total_runs += 1

            if last_step < threshold:
                converged_runs += 1
                curves_by_label_filtered[label].append((steps, ce_curve))
                misfrequency_by_label_filtered[label].append(misfrequency_by_label[label][i])
                variance_by_label_filtered[label].append(variance_by_label[label][i])
            else:
                curves_by_label_not_converged[label].append((steps, ce_curve))
                misfrequency_by_label_not_converged[label].append(misfrequency_by_label[label][i])
                variance_by_label_not_converged[label].append(variance_by_label[label][i])

    fraction_converged = converged_runs / total_runs
    fraction_not_converged = 1.0 - fraction_converged

    return (
        curves_by_label_filtered, 
        misfrequency_by_label_filtered, 
        variance_by_label_filtered,
        curves_by_label_not_converged, 
        misfrequency_by_label_not_converged, 
        variance_by_label_not_converged,
        fraction_converged, 
        fraction_not_converged
    )