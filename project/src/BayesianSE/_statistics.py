"""
Statistical Analysis and Visualization Module for Bayesian State Estimation.

This module provides tools to evaluate the performance of Bayesian state 
estimation runs. It includes functions to calculate the variance of discrete 
probability distributions and to perform a comprehensive statistical analysis 
of estimation results, including success/failure categorization and 
distribution plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Any, Optional

from ._run_manager import plot_bayesian_run


def compute_variance(posterior: np.ndarray) -> float:
    """
    Computes the variance of a discrete probability distribution.

    This function treats the indices of the posterior array as the values 
    of the random variable and calculates the variance based on their 
    assigned probabilities.

    Parameters
    ----------
    posterior : np.ndarray or list
        The probability distribution (posterior) to evaluate.

    Returns
    -------
    variance : float
        The calculated variance of the distribution.
    """
    posterior = np.asarray(posterior)
    x = np.arange(len(posterior))  

    mean = np.sum(x * posterior)
    mean_sq = np.sum((x**2) * posterior)
    variance = mean_sq - mean**2

    return variance


def compute_statistics(
    curves_by_label: Dict[str, List[Tuple[np.ndarray, np.ndarray]]], 
    num_updates: int, 
    block_steps: int, 
    plot: bool = True, 
    filename: str = "figure_statistics.png"
) -> Tuple[Dict[str, Dict[str, Any]], List[int], List[int], List[int]]:
    """
    Analyzes and visualizes the performance metrics of Bayesian runs.

    The function categorizes runs into "Success" or "Failure" based on a 
    Cross Entropy threshold (corresponding to a posterior probability > 0.9).
    It calculates descriptive statistics for both the final probabilities 
    and the number of steps required to converge.

    Parameters
    ----------
    curves_by_label : dict
        A dictionary mapping labels to lists of (steps, cross_entropy) pairs.
    num_updates : int
        Number of updates performed in the simulation.
    block_steps : int
        Number of steps per measurement block.
    plot : bool, optional
        If True, generates and displays diagnostic plots. Default is True.
    filename : str, optional
        Base filename for the saved plots. Default is "figure_statistics.png".

    Returns
    -------
    stats : dict
        A nested dictionary containing descriptive statistics for Success, 
        Failure, and Total runs.
    final_steps_all : list
        The number of steps for every run.
    final_steps_below : list
        The number of steps for successful runs.
    final_steps_above : list
        The number of steps for failed runs.
    """

    final_steps_below = []
    final_steps_above = []
    final_cross_entropy_below = []
    final_cross_entropy_above = []
    
    threshold = -np.log(0.9)

    for label, curves in curves_by_label.items():
        for steps, cross_entropy in curves:
            final_step = steps[-1]
            final_ce = cross_entropy[-1]
            if final_ce < threshold:
                final_steps_below.append(final_step)
                final_cross_entropy_below.append(final_ce)
            else:
                final_steps_above.append(final_step)
                final_cross_entropy_above.append(final_ce)

    final_steps_all = final_steps_below + final_steps_above
    final_cross_entropy_all = final_cross_entropy_below + final_cross_entropy_above

    final_cross_entropy_all = [np.exp(-ce) for ce in final_cross_entropy_all]
    final_cross_entropy_below = [np.exp(-ce) for ce in final_cross_entropy_below]
    final_cross_entropy_above = [np.exp(-ce) for ce in final_cross_entropy_above]

    prob_mean_success = prob_std_success = prob_median_success = None
    prob_q25_success = prob_q75_success = prob_iqr_success = None
    prob_mean_failure = prob_std_failure = prob_median_failure = None
    prob_q25_failure = prob_q75_failure = prob_iqr_failure = None
    prob_mean_all = prob_std_all = prob_median_all = None
    prob_q25_all = prob_q75_all = prob_iqr_all = None

    steps_mean_success = steps_std_success = steps_median_success = None
    steps_q25_success = steps_q75_success = steps_iqr_success = None
    steps_mean_failure = steps_std_failure = steps_median_failure = None
    steps_q25_failure = steps_q75_failure = steps_iqr_failure = None
    steps_mean_all = steps_std_all = steps_median_all = None
    steps_q25_all = steps_q75_all = steps_iqr_all = None

    # --- Print Probability Statistics ---
    print("------------------------------------")
    print("------PROBABILITY STATISTICS------")
    print("------------------------------------")

    print("SUCCESSFUL RUNS STATISTICS")
    if len(final_cross_entropy_below) == 0:
        print("No successful runs. Statistics unavailable.\n")
    else:
        prob_mean_success = np.mean(final_cross_entropy_below)
        prob_std_success = np.std(final_cross_entropy_below)
        prob_median_success = np.median(final_cross_entropy_below)
        prob_q25_success, prob_q75_success = np.percentile(final_cross_entropy_below, [25, 75])
        prob_iqr_success = prob_q75_success - prob_q25_success

        print(f"Mean value: {prob_mean_success:.4f}")
        print(f"Standard deviation: {prob_std_success:.4f}")
        print(f"Median: {prob_median_success:.4f}")
        print(f"IQR (interquartile range): {prob_iqr_success:.4f}", '\n')

    print("FAILURE RUNS STATISTICS")
    if len(final_cross_entropy_above) == 0:
        print("No failure runs. Statistics unavailable.\n")
    else:
        prob_mean_failure = np.mean(final_cross_entropy_above)
        prob_std_failure = np.std(final_cross_entropy_above)
        prob_median_failure = np.median(final_cross_entropy_above)
        prob_q25_failure, prob_q75_failure = np.percentile(final_cross_entropy_above, [25, 75])
        prob_iqr_failure = prob_q75_failure - prob_q25_failure

        print(f"Mean value: {prob_mean_failure:.4f}")
        print(f"Standard deviation: {prob_std_failure:.4f}")
        print(f"Median: {prob_median_failure:.4f}")
        print(f"IQR (interquartile range): {prob_iqr_failure:.4f}", '\n')

    print("TOTAL RUNS STATISTICS")
    if len(final_cross_entropy_all) == 0:
        print("No total runs. Statistics unavailable.\n")
    else:
        prob_mean_all = np.mean(final_cross_entropy_all)
        prob_std_all = np.std(final_cross_entropy_all)
        prob_median_all = np.median(final_cross_entropy_all)
        prob_q25_all, prob_q75_all = np.percentile(final_cross_entropy_all, [25, 75])
        prob_iqr_all = prob_q75_all - prob_q25_all

        print(f"Mean value: {prob_mean_all:.4f} ")
        print(f"Standard deviation: {prob_std_all:.4f} ")
        print(f"Median: {prob_median_all:.4f} ")
        print(f"IQR (interquartile range): {prob_iqr_all:.4f} ", '\n')

    # Define bins for step-count histograms
    bins = np.arange(0, max(final_steps_below + final_steps_above) + 10, 10)

    if plot:
        # Histograms of convergence speed for Success vs Failure
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(final_steps_below, bins=bins, alpha=0.6, color="#1f77b4",
                edgecolor='black', label="Success estimation")
        ax.hist(final_steps_above, bins=bins, alpha=0.6, color="#ff7f0e",
                edgecolor='black', label="Failure estimation")

        ax.set_xlabel("Number of steps", fontsize=20)
        ax.set_ylabel("Frequency", fontsize=20)
        ax.set_title("Distribution of the final step values", fontsize=25)
        ax.legend(fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad=8)

        name, ext = os.path.splitext(filename)
        filename_category = f"{name}_steps{ext}"
        fig.tight_layout()

        plot_bayesian_run(fig, filename_category)
        plt.show()

    # --- Print Step Statistics ---
    print("----------------------------")
    print("------STEPS STATISTICS------")
    print("----------------------------")

    print("SUCCESSFUL RUNS STATISTICS")
    if len(final_steps_below) == 0:
        print("No successful runs. Statistics unavailable.\n")
    else:
        steps_mean_success = np.mean(final_steps_below)
        steps_std_success = np.std(final_steps_below)
        steps_median_success = np.median(final_steps_below)
        steps_q25_success, steps_q75_success = np.percentile(final_steps_below, [25, 75])
        steps_iqr_success = steps_q75_success - steps_q25_success

        print(f"Mean value: {steps_mean_success:.1f} steps")
        print(f"Standard deviation: {steps_std_success:.1f} steps")
        print(f"Median: {steps_median_success:.1f} steps")
        print(f"IQR (interquartile range): {steps_iqr_success:.1f} steps", '\n')

    print("FAILURE RUNS STATISTICS")
    if len(final_steps_above) == 0:
        print("No failure runs. Statistics unavailable.\n")
    else:
        steps_mean_failure = np.mean(final_steps_above)
        steps_std_failure = np.std(final_steps_above)
        steps_median_failure = np.median(final_steps_above)
        steps_q25_failure, steps_q75_failure = np.percentile(final_steps_above, [25, 75])
        steps_iqr_failure = steps_q75_failure - steps_q25_failure

        print(f"Mean value: {steps_mean_failure:.1f} steps")
        print(f"Standard deviation: {steps_std_failure:.1f} steps")
        print(f"Median: {steps_median_failure:.1f} steps")
        print(f"IQR (interquartile range): {steps_iqr_failure:.1f} steps", '\n')

    print("TOTAL RUNS STATISTICS")
    if len(final_steps_all) == 0:
        print("No total runs. Statistics unavailable.\n")
    else:
        steps_mean_all = np.mean(final_steps_all)
        steps_std_all = np.std(final_steps_all)
        steps_median_all = np.median(final_steps_all)
        steps_q25_all, steps_q75_all = np.percentile(final_steps_all, [25, 75])
        steps_iqr_all = steps_q75_all - steps_q25_all

        print(f"Mean value: {steps_mean_all:.1f} steps")
        print(f"Standard deviation: {steps_std_all:.1f} steps")
        print(f"Median: {steps_median_all:.1f} steps")
        print(f"IQR (interquartile range): {steps_iqr_all:.1f} steps", '\n')

    max_steps = num_updates * block_steps

    # Data categorization for plotting KDE (Kernel Density Estimation)
    datasets = [
        ("Successes Only", final_steps_below, steps_mean_success, steps_std_success, steps_median_success),
        ("All Runs", final_steps_all, steps_mean_all, steps_std_all, steps_median_all)
    ]

    kde_color = "#2A2F92"
    highlight_color = "#D40035"
    median_color = '#0C630B'

    if plot:
        # Comparison plots showing density and spread
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, (title, data, mean_val, std_val, median_val) in zip(axes, datasets):
            # Plot the KDE for the convergence steps
            kde = sns.kdeplot(data, ax=ax, color=kde_color, linewidth=1.5)
            x, y = kde.get_lines()[0].get_data()

            # Shade the area corresponding to +/- 1 Standard Deviation
            mask = (x >= mean_val - std_val) & (x <= mean_val + std_val)
            ax.fill_between(x[mask], y[mask], color=highlight_color, alpha=0.3, 
                            label=f"±1σ [{mean_val - std_val:.1f}, {mean_val + std_val:.1f}]")

            # Annotate Mean and Median
            ax.axvline(mean_val, color=highlight_color, linestyle='-.', linewidth=1, 
                       label=f"Mean = {mean_val:.1f}")
            ax.axvline(median_val, color=median_color, linestyle='-.', linewidth=1, 
                       label=f"Median = {median_val:.1f}")

            ax.set_title(f"Steps Distribution\n({title})", fontsize=25)
            ax.set_xlabel("Number of steps", fontsize=20)
            ax.set_xlim(0, max_steps)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad=8)
            ax.legend(loc='upper right', fontsize=20)

        axes[0].set_ylabel("Density", fontsize=20)
        plt.tight_layout()
        plt.show()

    # Compile all statistics into a structured nested dictionary
    stats = {
        "probability": {
            "success": {
                "mean": prob_mean_success, "std": prob_std_success, 
                "median": prob_median_success, "iqr": prob_iqr_success,
                "q25": prob_q25_success, "q75": prob_q75_success,
            },
            "failure": {
                "mean": prob_mean_failure, "std": prob_std_failure,
                "median": prob_median_failure, "iqr": prob_iqr_failure,
                "q25": prob_q25_failure, "q75": prob_q75_failure,
            },
            "all": {
                "mean": prob_mean_all, "std": prob_std_all,
                "median": prob_median_all, "iqr": prob_iqr_all,
                "q25": prob_q25_all, "q75": prob_q75_all,
            }
        },
        "steps": {
            "success": {
                "mean": steps_mean_success, "std": steps_std_success,
                "median": steps_median_success, "iqr": steps_iqr_success,
                "q25": steps_q25_success, "q75": steps_q75_success,
            },
            "failure": {
                "mean": steps_mean_failure, "std": steps_std_failure,
                "median": steps_median_failure, "iqr": steps_iqr_failure,
                "q25": steps_q25_failure, "q75": steps_q75_failure,
            },
            "all": {
                "mean": steps_mean_all, "std": steps_std_all,
                "median": steps_median_all, "iqr": steps_iqr_all,
                "q25": steps_q25_all, "q75": steps_q75_all,
            }
        }
    }

    return stats, final_steps_all, final_steps_below, final_steps_above