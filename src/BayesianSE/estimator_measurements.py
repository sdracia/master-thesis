"""
Measurement Configuration and Sensitivity Analysis for Bayesian Estimation.

This module provides tools to:
- Define a library of measurement settings (pulses) for the Bayesian estimator.
- Perform marginalization over laser noise and miscalibration distributions.
- Validate likelihood matrices for physical consistency (normalization and positivity).
- Visualize the sensitivity of measurement pulses across different rotational manifolds.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Union, List, Any

from ._noise_models import build_detuning_distributions
from saving import save_figure_in_images


def measurement_setting(
    self, 
    rabi_by_j: Union[float, Dict[Tuple[int, int], float]], 
    dephased: bool, 
    coherence_time_us: float, 
    is_minus: bool, 
    noise_params: Optional[Dict[str, Dict[str, float]]] = None, 
    seed: Optional[int] = None, 
    max_excitation: float = 0.9,
    laser_miscalibration: Optional[Dict[str, Dict[str, float]]] = None,
    seed_miscalibration: Optional[int] = None,
    marginalization: bool = True
) -> None:
    """
    Configures the set of available measurement pulses for the Bayesian estimator.

    This function prepares a library of pulse parameters (frequency, duration, 
    Rabi rate) for each rotational manifold J. It calculates the corresponding 
    likelihood matrices (lh0, lh1), optionally integrating over noise distributions.

    Parameters
    ----------
    rabi_by_j : float or dict
        If float, a single Rabi rate is used for all J.
        If dict, maps (J_min, J_max) tuples to specific Rabi rates.
    dephased : bool
        Whether to model the excitation with dephasing.
    coherence_time_us : float
        Coherence time for the dephasing model in microseconds.
    is_minus : bool
        Initial transition direction (Delta_m = -1 if True).
    noise_params : dict, optional
        Shot-to-shot noise characteristics for frequency and Rabi rate.
    seed : int, optional
        Seed for noise generation.
    max_excitation : float, optional
        The maximum attainable excitation probability. Default is 0.9.
    laser_miscalibration : dict, optional
        Systematic laser calibration errors.
    seed_miscalibration : int, optional
        Seed for calibration error generation.
    marginalization : bool, optional
        If True, likelihoods are averaged over the frequency noise distribution.
        Default is True.
    """

    # --- 1. PRE-VALIDATION AND MARGINALIZATION LOGIC ---
    freq_mis_level = (laser_miscalibration or {}).get("frequency", {}).get("level", 0.0)
    freq_noise_level = (noise_params or {}).get("frequency", {}).get("level", 0.0)

    if freq_mis_level + freq_noise_level == 0.0:
        marginalization = False

    df_trans = self.model.transition_df

    # --- 2. CONSTRUCT MEASUREMENT PULSE LIBRARY ---
    if isinstance(rabi_by_j, dict):
        measurements = []
        for (j_min, j_max), rabi_rate_mhz in rabi_by_j.items():
            for j in range(j_min, j_max + 1):
                measurements.append([
                    j,
                    (-1 if not is_minus else 1) * df_trans.loc[df_trans["j"] == j].iloc[0]["energy_diff"] * 1e-3, 
                    np.pi / (rabi_rate_mhz * df_trans.loc[df_trans["j"] == j].iloc[0]["coupling"]), 
                    dephased, 
                    coherence_time_us,
                    is_minus,
                    rabi_rate_mhz
                ])
        
        is_minus = not is_minus

        for (j_min, j_max), rabi_rate_mhz in rabi_by_j.items():
            for j in range(j_min, j_max + 1):
                measurements.append([
                    j,
                    (-1 if not is_minus else 1) * df_trans.loc[df_trans["j"] == j].iloc[0]["energy_diff"] * 1e-3,
                    np.pi / (rabi_rate_mhz * df_trans.loc[df_trans["j"] == j].iloc[0]["coupling"]),
                    dephased,
                    coherence_time_us,
                    is_minus,
                    rabi_rate_mhz
                ])

    elif isinstance(rabi_by_j, (int, float)):
        measurements = [
            [
                j,
                (-1 if not is_minus else 1) * df_trans.loc[df_trans["j"] == j].iloc[0]["energy_diff"] * 1e-3, 
                np.pi / (rabi_by_j * df_trans.loc[df_trans["j"] == j].iloc[0]["coupling"]), 
                dephased, 
                coherence_time_us,
                is_minus,
                rabi_by_j
            ] for j in range(1, self.j_max + 1)
        ]

        is_minus = not is_minus
        
        measurements += [
            [
                j,
                (-1 if not is_minus else 1) * df_trans.loc[df_trans["j"] == j].iloc[0]["energy_diff"] * 1e-3,
                np.pi / (rabi_by_j * df_trans.loc[df_trans["j"] == j].iloc[0]["coupling"]),
                dephased,
                coherence_time_us,
                is_minus,
                rabi_by_j
            ] for j in range(1, self.j_max + 1)
        ]
    else:
        raise TypeError("rabi_rate_mhz must be a float or a dict mapping (j_min, j_max) to rates.")
    
    self.Probs_exc_list = []

    # --- 3. COMPUTE LIKELIHOOD MATRICES ---
    if marginalization:
        for j, frequency, duration, deph, coh_time, is_min, rabi in measurements:
            freq_grid, freq_probs = build_detuning_distributions(
                frequency=frequency,
                rabi_rate=rabi,
                laser_miscalibration=laser_miscalibration,
                noise_params=noise_params
            )

            lh0, lh1 = 0.0, 0.0

            for f, p in zip(freq_grid, freq_probs):
                l0, l1 = self.likelihoods_estimator(
                    frequency=f,
                    duration_us=duration,
                    rabi_rate_mhz=rabi,
                    dephased=deph,
                    coherence_time_us=coh_time,
                    is_minus=is_min,
                    noise_params=None, 
                    seed=None,
                    maximum_excitation=max_excitation,
                    laser_miscalibration=None,
                    seed_miscalibration=None
                )
                lh0 += p * l0
                lh1 += p * l1

            exc_mat = lh0 + lh1

            tol_neg = 1e-6  
            if np.any(exc_mat.data < -tol_neg):
                raise ValueError("Significant negative values found in likelihood matrix.")
            
            row_sums = np.array(exc_mat.sum(axis=0)).flatten()
            if np.any(np.abs(row_sums - 1.0) > 1e-3):
                raise ValueError("The likelihood matrix is not properly normalized.")

            self.Probs_exc_list.append((lh0, lh1))

    else: 
        for _, frequency, duration, deph, coh_time, is_min, rabi in measurements:
            lh0, lh1 = self.likelihoods_estimator(
                frequency=frequency,
                duration_us=duration,
                rabi_rate_mhz=rabi,
                dephased=deph,
                coherence_time_us=coh_time,
                is_minus=is_min,
                noise_params=noise_params,
                seed=seed,
                maximum_excitation=max_excitation,
                laser_miscalibration=laser_miscalibration,
                seed_miscalibration=seed_miscalibration
            )
            self.Probs_exc_list.append((lh0, lh1))

    self.measurements = measurements


def meas_sensitivity_heatmap(
    Estimator: Any, 
    final_index: np.ndarray, 
    initial_index: np.ndarray, 
    figname: str = "meas_sensitivity.svg", 
    title: Optional[str] = None
) -> None:
    """
    Generates a heatmap showing the sensitivity of measurement pulses across manifolds.

    The sensitivity is defined as the transition probability (excitation likelihood)
    induced by a specific measurement pulse on a given rotational state J.

    Parameters
    ----------
    Estimator : BayesianStateEstimation
        The estimator instance containing calculated probabilities.
    final_index : np.ndarray
        Array mapping J manifolds to final state indices.
    initial_index : np.ndarray
        Array mapping J manifolds to initial state indices.
    figname : str, optional
        Filename for the saved plot.
    title : str, optional
        Custom title for the heatmap.
    """
    measurements = Estimator.measurements
    probabilities = Estimator.Probs_exc_list
    j_max = Estimator.j_max

    heatmap_data = np.zeros((j_max, j_max))

    for meas_idx in range(j_max):
        _, lh_matrix = probabilities[meas_idx]
        for j in range(1, j_max + 1):
            value = lh_matrix[final_index[j], initial_index[j]]
            heatmap_data[meas_idx, j - 1] = value 

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Excitation probability", fontsize=25)
    cbar.ax.tick_params(labelsize=18)

    ax.set_xlabel(r"$J$ rotational manifold", fontsize=24)
    ax.set_ylabel(r"J measurement pulse $\mu_J^{+}$", fontsize=24)
    if title is None:
        title = "Measurement sensitivity"

    ax.set_title(title, fontsize=28)
    
    tick_range = np.arange(0, j_max, 5)
    tick_labels = np.arange(1, j_max + 1, 5)
    ax.set_xticks(tick_range)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_range)
    ax.set_yticklabels(tick_labels)

    ax.tick_params(axis='both', pad=10, labelsize=20)

    plt.tight_layout()
    save_figure_in_images(fig, filename=figname)
    plt.show()