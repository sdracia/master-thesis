# File: _plotting.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Visualization Module for Bayesian State Estimation and Simulation.

This module provides tools for:
- Mapping molecular state distributions into 2D grid matrices for visualization.
- Generating animations of posterior evolution compared to ground-truth states.
- Statistical plotting of Cross-Entropy convergence and category-based performance.
- Comparative analysis between different estimation methods using error-clipped plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

from ._run_manager import plot_bayesian_run
from saving import save_figure_in_images


def matrix_creation(
    df: pd.DataFrame, 
    posterior: np.ndarray, 
    j_max: int, 
    normalize: bool = True
) -> np.ndarray:
    """
    Transforms a posterior vector into a 2D matrix organized by J and m quantum numbers.

    The function maps the internal states into a grid where columns represent J 
    and rows represent m. It handles parity (xi) by splitting each grid cell 
    into two values.

    Parameters
    ----------
    df : pd.DataFrame
        The molecular state dataframe.
    posterior : np.ndarray
        The current probability distribution vector.
    j_max : int
        Maximum rotational quantum number.
    normalize : bool, optional
        If True, renormalizes the population within each J manifold. Default is True.

    Returns
    -------
    matrix : np.ndarray
        A 2D array of objects (lists [xi_false_val, xi_true_val]) or NaNs.
    """
    dataframe = df.copy()
    dataframe["posterior"] = posterior

    if normalize:
        for j in range(j_max + 1):
            states_in_j = dataframe.loc[dataframe["j"] == j]
            state_dist = states_in_j["posterior"].to_numpy()
            state_dist = state_dist / np.sum(state_dist)
            dataframe.loc[dataframe["j"] == j, "posterior"] = state_dist

    # Group by J and m, aggregating the two parity (xi) states into a list
    df_grouped = dataframe.groupby(['j', 'm']).agg(
        {'xi': 'first', 'posterior': list}
    ).reset_index()
    
    # Ensure lists represent [xi=False, xi=True] even if one state is missing
    for index, row in df_grouped.iterrows():
        if len(row['posterior']) == 1:
            if row['xi'] == False:
                df_grouped.at[index, 'posterior'] = [row['posterior'][0], np.nan]
            elif row['xi'] == True:
                df_grouped.at[index, 'posterior'] = [np.nan, row['posterior'][0]]
    
    df_coords = df_grouped[["j", "m"]]
    state_vals = df_grouped["posterior"].tolist()
    
    # Initialize grid: height spans 2*j_max + 1, width spans j_max + 1
    sq_array = np.zeros((2 * (j_max + 1), j_max + 1), dtype=object)
    sq_array[:] = np.nan
    
    list_index = []
    for _, row in df_coords.iterrows():
        j = row['j']
        m = row['m']
        # Map m to positive row indices
        list_index.append([int(j_max + m + 0.5), int(j)])
    
    for data_idx, idx_tuple in enumerate(list_index):
        sq_array[idx_tuple[0], idx_tuple[1]] = state_vals[data_idx]
    
    return sq_array


def simulator_state(
    df: pd.DataFrame, 
    index: int, 
    j_max: int
) -> List[float]:
    """
    Determines the grid coordinates of the physical simulator state.

    Parameters
    ----------
    df : pd.DataFrame
        Molecular state dataframe.
    index : int
        The global index of the current state.
    j_max : int
        Maximum rotational quantum number.

    Returns
    -------
    coordinates : list
        List of [row_index, column_offset] for visualization mapping.
    """
    dataframe = df.copy()
    dataframe["outcome"] = 0
    dataframe.loc[index, "outcome"] = 1

    df_grouped = dataframe.groupby(['j', 'm']).agg(
        {'xi': 'first', 'outcome': list}
    ).reset_index()
    
    for idx_row, row in df_grouped.iterrows():
        if len(row['outcome']) == 1:
            if row['xi'] == False:
                df_grouped.at[idx_row, 'outcome'] = [row['outcome'][0], np.nan]
            elif row['xi'] == True:
                df_grouped.at[idx_row, 'outcome'] = [np.nan, row['outcome'][0]]
    
    df_coords = df_grouped[["j", "m"]]
    state_vals = df_grouped["outcome"].tolist()
    
    sq_array = np.zeros((2 * (j_max + 1), j_max + 1), dtype=object)
    sq_array[:] = np.nan
    
    list_index = []
    for _, row in df_coords.iterrows():
        j = row['j']
        m = row['m']
        list_index.append([int(j_max + m + 0.5), int(j)])
    
    for data_idx, idx_tuple in enumerate(list_index):
        sq_array[idx_tuple[0], idx_tuple[1]] = state_vals[data_idx]

    coordinates = [0.0, 0.0]
    # Identify where the "1" (active state) is located
    for (x, y), w in np.ndenumerate(sq_array):
        if isinstance(w, float):
            continue
        else:
            w_false = w[0]
            w_true = w[1]
            if w_false == 1:
                coordinates = [x, y + 0.25]   
            if w_true == 1:
                coordinates = [x, y - 0.25]

    return coordinates 


def plot_matrix(
    ax: plt.Axes, 
    matrix: np.ndarray, 
    coordinate_out: List[float], 
    j_max: int
) -> None:    
    """
    Renders the Bayesian belief matrix and marks the simulator ground truth.

    Belief magnitude is shown as square sizes (blue), while the physical 
    state is highlighted with a red border.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to draw on.
    matrix : np.ndarray
        Belief grid from matrix_creation.
    coordinate_out : list
        Physical state coordinates from simulator_state.
    j_max : int
        Max J value for axis centering.
    """
    max_weight = 1
    ax_bkgdcolor = "white"
    ax_facecolor = '#D3D3D3'
    
    for (x, y), w in np.ndenumerate(matrix):
        if isinstance(w, float):
            # Background placeholder for empty cells
            size = 1.0
            rect = plt.Rectangle(
                [x - 0.5 - size / 2 - j_max, y - size / 2],
                size, size, facecolor=ax_bkgdcolor, edgecolor=ax_bkgdcolor
            )
            ax.add_patch(rect)
        else:
            w_false, w_true = w[0], w[1]
    
            # Plot state with xi = False
            if not np.isnan(w_false):
                face_color, edge_color = "blue", "blue"
                w_plot = abs(w_false)
                size = np.sqrt(w_plot / max_weight)
            else:
                size, face_color, edge_color = 1.0, ax_bkgdcolor, ax_bkgdcolor
    
            rect_f = plt.Rectangle(
                [x - 0.5 - size / 2 - j_max, y + 0.25 - (size * 0.5) / 2],
                size, size * 0.5, facecolor=face_color, edgecolor=edge_color
            )
            ax.add_patch(rect_f)
    
            # Plot state with xi = True
            if not np.isnan(w_true):
                face_color, edge_color = "blue", "blue"
                w_plot = abs(w_true)
                size = np.sqrt(w_plot / max_weight)
            else:
                size, face_color, edge_color = 1.0, ax_bkgdcolor, ax_bkgdcolor
    
            rect_t = plt.Rectangle(
                [x - 0.5 - size / 2 - j_max, y - 0.25 - (size * 0.5) / 2],
                size, size * 0.5, facecolor=face_color, edgecolor=edge_color
            )
            ax.add_patch(rect_t)
            
    # Mark Simulator Ground Truth
    x_out, y_out = coordinate_out[0], coordinate_out[1]
    rect_out = plt.Rectangle(
        [x_out - 0.5 - 1 / 2 - j_max, y_out - 0.5 / 2],
        1, 0.5, facecolor='none', edgecolor="red"
    )
    ax.add_patch(rect_out)
    
    ax.patch.set_facecolor(ax_facecolor)
    ax.set_aspect("equal", "box")
    ax.set_ylim([-0.5, matrix.shape[1] - 0.5])


def heatmap_posterior_animation(
    Estimator: Any, 
    Simulator: Any, 
    molecule: Any, 
    j_max: int, 
    folder: str = "animations", 
    base_filename: str = "evolution_EwS_"
) -> None:
    """
    Creates a GIF showing the evolution of the estimator belief vs physical state.
    """
    Estimator_list = Estimator.history_list
    Simulator_list = Simulator.history_list

    posteriors = [np.array(entry["posterior"]) for entry in Estimator_list]
    matrices = [
        matrix_creation(df=molecule.state_df, posterior=p, j_max=j_max, normalize=False)
        for p in posteriors
    ]

    coordinates = []
    for i in range(len(Simulator_list[1:])):
        idx = Simulator_list[i + 1][0]
        coord = simulator_state(df=molecule.state_df, index=idx, j_max=j_max)
        coordinates.append(coord)

    fig, ax = plt.subplots(figsize=(20, 13))
    ax_facecolor = '#D3D3D3'

    def update(step):
        ax.clear()
        ax.set_facecolor(ax_facecolor)
        matrix = matrices[step]
        coordinate_out = coordinates[step]
        plot_matrix(ax, matrix, coordinate_out, j_max)

        # Axis labeling and formatting
        ax.set_xlim([-j_max - 1, j_max + 1])
        ax.set_yticks(np.arange(0, j_max + 1))
        ax.set_yticks(np.arange(0, j_max + 2) - 0.5, minor=True)
        ax.grid(True, which="minor", color="w")
        ax.set_xlabel("$m$")
        ax.set_ylabel("$J$")
        ax.set_xticks(np.arange(-j_max - 0.5, j_max + 1.5, 1))
        ax.set_title(f"Evolution of Estimator Posterior with Simulator state (Step {step})")

    ani = FuncAnimation(fig, update, frames=len(Estimator_list), interval=500)
    
    # Save GIF with automatic numbering
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.startswith(base_filename) and f.endswith(".gif")]
    nums = [int(f.split("_")[-1].split(".")[0]) for f in existing if f.split("_")[-1].split(".")[0].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    filename = os.path.join(folder, f"{base_filename}_{next_num}.gif")

    ani.save(filename, writer="ffmpeg", fps=2)
    print(f"Animation saved to: {filename}")


def vector_posterior_animation(
    Estimator: Any, 
    folder: str = "animations", 
    base_filename: str = "evolution_plot"
) -> None:
    """
    Creates a GIF showing the Prior and Posterior evolution as linear vectors.
    """
    num_steps = len(Estimator.history_list)
    fig, ax = plt.subplots(figsize=(10, 4))
    line_prior, = ax.plot([], [], linestyle="-", color="blue", label="Prior")
    line_posterior, = ax.plot([], [], linestyle="-.", color="green", label="Posterior")

    ax.set_xlabel("State Index")
    ax.set_ylabel("Probability Value")
    ax.legend()
    ax.grid(True)
    
    xlim = len(Estimator.history_list[0]["prior"])
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 1)  

    def update(step):
        entry = Estimator.history_list[step]
        prior = np.array(entry["prior"])
        posterior = np.array(entry["posterior"])
        line_prior.set_data(range(len(prior)), prior)
        line_posterior.set_data(range(len(posterior)), posterior)
        ax.set_title(f"Bayesian Evolution: Prior vs Posterior (Step {step})")
        return line_prior, line_posterior

    ani = FuncAnimation(fig, update, frames=num_steps, interval=200, blit=False)
    
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.startswith(base_filename) and f.endswith(".gif")]
    nums = [int(f.split("_")[-1].split(".")[0]) for f in existing if f.split("_")[-1].split(".")[0].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    filename = os.path.join(folder, f"{base_filename}_{next_num}.gif")

    ani.save(filename, writer="ffmpeg", fps=2)
    print(f"Animation saved to: {filename}")


def plot_cross_entropies_vs_step(
    curves_by_label: Dict[str, list], 
    color_map: Dict[str, str], 
    filename: str = "figure.png"
) -> None:
    """
    Plots Cross-Entropy curves vs simulation steps categorized by label.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    seen_labels = set()

    for label, curves in curves_by_label.items():
        for steps, ce in curves:
            ax.plot(steps, ce, linestyle='-', color=color_map[label], alpha=0.1,
                    label=label if label not in seen_labels else "")
            seen_labels.add(label)

    ax.set_xlabel("Step", fontsize=20)
    ax.set_ylabel("Cross-Entropy", fontsize=20)
    ax.set_title("Monte Carlo Initialization: Cross-Entropy vs Step", fontsize=25)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', labelsize=20, pad=8)

    fig.tight_layout()
    plot_bayesian_run(fig, filename)
    plt.show()

    # Create category-specific subplots
    name, ext = os.path.splitext(filename)
    filename_category = f"{name}_per_category{ext}"

    fig_cat, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    label_order = ['LM', 'PU', 'PL', 'Other']

    for ax_cat, label in zip(axs.flat, label_order):
        for steps, ce in curves_by_label[label]:
            ax_cat.plot(steps, ce, linestyle='-', color=color_map[label], alpha=0.2)
        ax_cat.set_title(f"{label}")
        ax_cat.grid(True, linestyle='--', alpha=0.5)
        ax_cat.set_xlabel("Step", fontsize=20)
        ax_cat.set_ylabel("Cross-Entropy", fontsize=20)
        ax_cat.tick_params(axis='both', labelsize=20, pad=8)

    fig_cat.suptitle("Cross-Entropy vs Step per Category", fontsize=25)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_bayesian_run(fig_cat, filename_category)
    plt.show()

    total_elements = sum(len(lst) for lst in curves_by_label.values())
    print("Total number of initializations:", total_elements)

    for label, lst in curves_by_label.items():
        print(f"{label}: {len(lst)} Initializations")




def histo_final_cross_entropy(
    curves_by_label: Dict[str, list], 
    color_map: Dict[str, str], 
    only_total: bool = False, 
    filename: str = "figure_histo.png"
) -> Tuple[int, float, int, float]:
    """
    Plots histograms of the final Cross-Entropy values achieved by the estimator.
    """
    label_order = ['LM', 'PU', 'PL', 'Other']

    all_last_values = []
    curves_last_values = {}


    for label in label_order:
        last_values = [ce[-1] for _, ce in curves_by_label[label] if len(ce) > 0]
        last_values = np.array(last_values)
        negativi = last_values[last_values <= 0]
        if len(negativi) > 0:
            print(f"Negative values or zero for {label}: {negativi}")
        all_last_values.extend(last_values)
        curves_last_values[label] = last_values

    global_min = min(all_last_values)
    global_max = max(all_last_values)

    bins = np.logspace(np.log10(global_min), np.log10(global_max), 100)
    vline = -np.log(0.9)

    if not only_total:
        fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True)

        name, ext = os.path.splitext(filename)
        filename_category = f"{name}_per_category{ext}"

        for ax, label in zip(axs.flat, label_order):
            last_values = curves_last_values[label]

            ax.hist(last_values, bins=bins, color=color_map[label], alpha=0.7, edgecolor='black')
            ax.set_xscale('log')
            ax.axvline(vline, color='red', linestyle='--', label='H @ p=0.9')
            ax.set_title(f"{label} - Final cross-entropy", fontsize=25)
            ax.set_ylabel("Count", fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.5, which='both')
            ax.legend(fontsize=20)
            ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad = 8)

        for ax in axs[0]:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
            ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad = 8)


        fig.supxlabel("Cross-entropy", fontsize=20)
        fig.suptitle("Histogram of final Cross-entropy values per category", fontsize=25)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plot_bayesian_run(fig, filename_category)

        plt.show()


    fig, ax = plt.subplots(figsize=(8, 4))

    for label in label_order:
        last_values = curves_last_values[label]
        ax.hist(last_values, bins=bins, alpha=0.6, label=label,
                color=color_map[label], edgecolor='black')

    ax.axvline(vline, color='red', linestyle='--', label='H @ p=0.9')
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.5, which='both')
    ax.set_xlabel("Cross-entropy", fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.set_title("Combined histogram of final cross-entropy values by category", fontsize=25)
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad = 8)


    fig.tight_layout()

    plot_bayesian_run(fig, filename)

    plt.show()


    count_less = 0
    count_greater = 0

    for values in curves_last_values.values():
        count_less += np.sum(values < vline)
        count_greater += np.sum(values >= vline)

    print(f"Number of success events (H < -log(0.9) ): {count_less}")
    print(f"Number of failure events (H >= -log(0.9) ): {count_greater}", '\n')

    total_runs = sum(len(v) for v in curves_last_values.values())
    success_rate_total = count_less / total_runs

    leftmost_values = curves_last_values['LM']
    leftmost_successes = np.sum(leftmost_values < vline)
    leftmost_total = len(leftmost_values)
    success_rate_leftmost = leftmost_successes / leftmost_total if leftmost_total > 0 else np.nan

    print(f"Total initializations: {total_runs}")
    print(f"Total success probability: {success_rate_total:.3f}")
    print(f"Leftmost initializations: {leftmost_total}")
    print(f"Success probability given Leftmost: {success_rate_leftmost:.3f}")

    return total_runs, success_rate_total, leftmost_total, success_rate_leftmost




def plot_with_clipped_errors(
    ax: plt.Axes, 
    x: np.ndarray, 
    mean: list, 
    std: list, 
    median: list, 
    title: str, 
    N: int, 
    num_methods: int, 
    colors: list, 
    method_names: list, 
    success_rate_total: Optional[list] = None
) -> None:
    """
    Internal utility to plot error bars clipped to physical bounds (e.g., 0 to 1).
    """
    for i in range(num_methods):
        lower = max(0, mean[i] - std[i])


        if title == "Probability":
            upper = min(1, mean[i] + std[i])
        else:  
            upper = mean[i] + std[i]

        yerr = np.array([[mean[i] - lower], [upper - mean[i]]])

        if success_rate_total is not None:
            std_succ = np.sqrt(success_rate_total[i] * (1 - success_rate_total[i]) / N) 
            lower_succ = max(0, success_rate_total[i] - std_succ)
            upper_succ = min(1, success_rate_total[i] + std_succ)

            yerr_succ = np.array([[success_rate_total[i] - lower_succ], [upper_succ - success_rate_total[i]]])

        ax.errorbar(x[i], mean[i], yerr=yerr, fmt='s',  
                    color=colors[i], capsize=14, markersize=8, elinewidth=1.5, capthick=2, markeredgecolor='black')

        ax.plot(x[i], median[i], marker='^',  
                color=colors[i], markersize=5, linestyle='None', markeredgecolor='black')
        
        if success_rate_total is not None:
            ax.errorbar(x[i], success_rate_total[i], yerr=yerr_succ, fmt='o',  
                    color=colors[i], capsize=8, markersize=5, elinewidth=1, markeredgecolor='black')
            cross_patch = plt.Line2D([0], [0], marker='o', color='black', markersize=5, linestyle='None', label="Success Rate")

    ax.set_title(title)
    ax.set_xticks(x)
    

    ax.set_xticklabels(method_names, rotation=45)
    # ax.set_yscale('log')
    ax.grid(True, linestyle="--", alpha=0.5)
    if title == "Probability":
        ax.axhline(0.9, color='red', linestyle='--', label='p=0.9')

    triangle_patch = plt.Line2D([0], [0], marker='^', color='black', markersize=5, linestyle='None', label="Median")

    square_patch = plt.Line2D([0], [0], marker='s', color='black', markersize=8,
                          linestyle='None', label="Mean")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)

    square_patch = plt.Line2D([0], [0], marker='s', color='black', markersize=8,
                              linestyle='None', label="Mean")            
    triangle_patch = plt.Line2D([0], [0], marker='^', color='black', markersize=6,
                                linestyle='None', label="Median")       

    handles = [square_patch, triangle_patch]

    if success_rate_total is not None:
        cross_patch = plt.Line2D([0], [0], marker='o', color='black', markersize=6,
                                 linestyle='None', label="Success Rate")  
        handles.append(cross_patch)

    existing_handles = ax.get_legend_handles_labels()[0]

    ax.legend(handles=handles + existing_handles, fontsize=10)



def plot_method_comparison(
    stats_list: list, 
    method_names: list, 
    success_rate_total: list, 
    N: int, 
    filename: str = "method_comparison.svg"
) -> None:
    """
    Generates high-level plots comparing different Bayesian estimation methods.
    """
    num_methods = len(stats_list)
    x = np.arange(num_methods)

    prob_mean_success = [s["probability"]["success"]["mean"] for s in stats_list]
    prob_std_success = [s["probability"]["success"]["std"] for s in stats_list]
    prob_median_success = [s["probability"]["success"]["median"] for s in stats_list]

    prob_mean_all = [s["probability"]["all"]["mean"] for s in stats_list]
    prob_std_all = [s["probability"]["all"]["std"] for s in stats_list]
    prob_median_all = [s["probability"]["all"]["median"] for s in stats_list]


    steps_mean_success = [s["steps"]["success"]["mean"] for s in stats_list]
    steps_std_success = [s["steps"]["success"]["std"] for s in stats_list]
    steps_median_success = [s["steps"]["success"]["median"] for s in stats_list]

    steps_mean_all = [s["steps"]["all"]["mean"] for s in stats_list]
    steps_std_all = [s["steps"]["all"]["std"] for s in stats_list]
    steps_median_all = [s["steps"]["all"]["median"] for s in stats_list]

    colors = plt.cm.tab10.colors[:num_methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("Method comparison \n Distribution of the final probability of correct estimation and number of steps", fontsize=14)

    plot_with_clipped_errors(axes[0], x, prob_mean_all, prob_std_all, prob_median_all, "Probability", N, num_methods, colors, method_names, success_rate_total=success_rate_total)
    axes[0].set_ylabel("Probability")

    plot_with_clipped_errors(axes[1], x, steps_mean_all, steps_std_all, steps_median_all, "Steps", N, num_methods, colors, method_names)
    axes[1].set_ylabel("Steps")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure_in_images(fig, filename)

    plt.show()

