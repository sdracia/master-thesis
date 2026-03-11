# File: _noise_models.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Module for Noise Modeling and Bayesian Performance Visualization.

This module provides utilities to:
- Calculate standard deviations for different noise models (Gaussian, Uniform).
- Construct frequency distributions for Bayesian marginalization.
- Visualize the final distributions of estimator variance and frequency 
  miscalibration across successful and failed simulation runs.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Optional, Any

from ._run_manager import plot_bayesian_run


def compute_sigma(
    source: Optional[Dict[str, Any]], 
    key: str, 
    value: float
) -> float:
    """
    Calculates the standard deviation (sigma) based on the specified noise model.

    Parameters
    ----------
    source : dict, optional
        A dictionary containing noise settings (type and level).
    key : str
        The specific parameter to look up (e.g., 'frequency' or 'rabi_rate').
    value : float
        The nominal value of the parameter, used for relative noise scaling.

    Returns
    -------
    float
        The calculated standard deviation. Returns 0.0 if the type is unknown.
    """
    entry = (source or {}).get(key, {})
    sigma_type = entry.get("type")
    level = entry.get("level", 0.0)

    if sigma_type == "abs_gaussian":
        return level

    elif sigma_type == "rel_gaussian":
        return level * abs(value)

    elif sigma_type == "abs_uniform":
        return level / np.sqrt(3)

    elif sigma_type == "rel_uniform":
        return (level * abs(value)) / np.sqrt(3)
    
    else:
        return 0.0


def build_detuning_distributions(
    frequency: float,
    rabi_rate: float,
    laser_miscalibration: dict,
    noise_params: dict,
    num_points: int = 1000,
    num_sigma: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a discretized frequency distribution for Bayesian marginalization.

    Even though experimental imperfections can affect multiple parameters, 
    marginalization is performed solely on the frequency axis as it is 
    typically the dominant source of estimation error.

    Parameters
    ----------
    frequency : float
        The central Raman frequency.
    rabi_rate : float
        The peak Rabi frequency.
    laser_miscalibration : dict
        Systematic miscalibration parameters.
    noise_params : dict
        Shot-to-shot noise parameters.
    num_points : int, optional
        Number of points in the discretized grid. Default is 1000.
    num_sigma : float, optional
        The range of the grid in units of total standard deviation. Default is 5.0.

    Returns
    -------
    freq_grid : np.ndarray
        The array of frequency points.
    freq_probs : np.ndarray
        The probability weights for each grid point, normalized to 1.
    """
    sigma_freq = np.sqrt(
        compute_sigma(laser_miscalibration, "frequency", frequency)**2 +
        compute_sigma(noise_params, "frequency", frequency)**2
    )

    freq_grid = np.linspace(
        frequency - num_sigma * sigma_freq,
        frequency + num_sigma * sigma_freq,
        num_points
    )
    
    dx_freq = freq_grid[1] - freq_grid[0]
    freq_pdf = norm.pdf(freq_grid, loc=frequency, scale=sigma_freq)
    
    freq_probs = freq_pdf * dx_freq
    freq_probs /= freq_probs.sum()  

    return freq_grid, freq_probs


def var_misfreq(
    variance_by_label: Dict[str, Any], 
    misfreq_by_label: Dict[str, Any], 
    filename: str = "figure_variance_misfreq.svg"
) -> None:
    """
    Generates histograms for the final variance and frequency miscalibration.

    The data is categorized into 'Success' and 'Failure' based on a 
    Cross Entropy threshold (-log(0.9)), which corresponds to a 
    90% posterior probability of correctly identifying the state.

    Parameters
    ----------
    variance_by_label : dict
        Dictionary of variance values at the end of simulation runs.
    misfreq_by_label : dict
        Dictionary of frequency miscalibration values encountered.
    filename : str, optional
        The output filename. Default is "figure_variance_misfreq.svg".
    """
    final_variance_below = []
    final_variance_above = []
    threshold = -np.log(0.9)

    for label, curves in variance_by_label.items():
        for last_var, cross_entropy in curves:
            if cross_entropy < threshold:
                final_variance_below.append(last_var)
            else:
                final_variance_above.append(last_var)

    all_vars = final_variance_below + final_variance_above
    max_v = np.max(all_vars)
    min_v = np.min(all_vars)
    bins_v = np.arange(min_v, max_v, (max_v - min_v) / 100)

    name, ext = os.path.splitext(filename)
    filename_var = f"{name}_variance{ext}"

    fig_v, ax_v = plt.subplots(figsize=(7, 4))
    ax_v.hist(final_variance_below, bins=bins_v, alpha=0.6, color="#1f77b4",
              edgecolor='black', label="Success estimation")
    ax_v.hist(final_variance_above, bins=bins_v, alpha=0.6, color="#ff7f0e",
              edgecolor='black', label="Failure estimation")

    ax_v.set_xlabel("Variance", fontsize=20)
    ax_v.set_ylabel("Frequency", fontsize=20)
    ax_v.set_title("Distribution of the final variance values", fontsize=25)
    ax_v.legend(fontsize=20)
    ax_v.grid(True, linestyle='--', alpha=0.4)
    ax_v.tick_params(axis='both', labelsize=20, pad=8)
    fig_v.tight_layout()

    plot_bayesian_run(fig_v, filename_var)
    plt.show()

    final_misfreq_below = []
    final_misfreq_above = []

    for label, curves in misfreq_by_label.items():
        for last_mis, cross_entropy in curves:
            if cross_entropy < threshold:
                final_misfreq_below.append(last_mis)
            else:
                final_misfreq_above.append(last_mis)

    all_mis = final_misfreq_below + final_misfreq_above
    max_m = np.max(all_mis)
    min_m = np.min(all_mis)

    if max_m == min_m:
        print("Simulation without miscalibration on frequency")
    else:   
        bins_m = np.arange(min_m, max_m, (max_m - min_m) / 100)

        fig_m, ax_m = plt.subplots(figsize=(7, 4))
        ax_m.hist(final_misfreq_below, bins=bins_m, alpha=0.6, color="#1f77b4",
                  edgecolor='black', label="Success estimation")
        ax_m.hist(final_misfreq_above, bins=bins_m, alpha=0.6, color="#ff7f0e",
                  edgecolor='black', label="Failure estimation")

        ax_m.set_xlabel("Raman frequency miscalibration (Hz)", fontsize=20)
        ax_m.set_ylabel("Frequency", fontsize=20)
        ax_m.set_title("Distribution of the miscalibration values", fontsize=25)
        ax_m.legend(fontsize=20)
        ax_m.grid(True, linestyle='--', alpha=0.4)
        ax_m.tick_params(axis='both', labelsize=20, pad=8)
        fig_m.tight_layout()

        filename_mis = f"{name}_misfreq{ext}"
        plot_bayesian_run(fig_m, filename_mis)
        plt.show()