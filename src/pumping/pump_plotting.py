"""
Visualization Module for Molecular and Motional Dynamics.

This module provides high-level plotting functions to visualize the temporal 
evolution of internal state populations and motional excitations in 
molecule-atom systems, supporting both single-stage and multi-stage simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from typing import Union, List, Optional, Any
from saving import save_figure_in_images


def plot_internal_and_motional_dynamics(
    results: Union[Result, List[Any]], 
    times: Optional[np.ndarray] = None, 
    n_internal: Optional[int] = None, 
    n_motional: Optional[int] = None, 
    title: Optional[str] = None, 
    savetext: Optional[str] = None, 
    only_pop: bool = False
) -> None:
    """
    Plots the internal state population heatmap and mean motional excitation.

    Parameters
    ----------
    results : Union[Result, List]
        Either a single QuTiP Result object or a list of [Result, times] pairs 
        representing sequential simulation stages.
    times : np.ndarray, optional
        Array of time points. Only required if 'results' is a single Result object.
    n_internal : int, optional
        Number of internal molecular states to be plotted on the y-axis.
    n_motional : int, optional
        Dimension of the motional Hilbert space for phonon number calculation.
    title : str, optional
        Custom title for the generated plots.
    savetext : str, optional
        If provided, the figure will be saved to the images directory with this filename.
    only_pop : bool, optional
        If True, only the internal population heatmap is displayed. Default is False.

    Returns
    -------
    None
    """

    # --- MULTI-STAGE CASE HANDLING ---
    if isinstance(results, list):
        all_times = []
        all_populations = []
        all_mean_n = []
        time_changes = []

        current_time_offset = 0.0

        for idx, (res, t) in enumerate(results):
            populations_stage = np.zeros((n_internal, len(t)))
            mean_n_stage = []

            n_op = tensor(qeye(n_internal), num(n_motional))

            for i, state in enumerate(res.states):
                for j in range(n_internal):
                    Pj_op = tensor(
                        basis(n_internal, j) * basis(n_internal, j).dag(), 
                        qeye(n_motional)
                    )
                    populations_stage[j, i] = expect(Pj_op, state)
                
                mean_n_stage.append(expect(n_op, state))

            progressive_time = current_time_offset + t

            if idx < len(results) - 1:
                populations_stage = populations_stage[:, :-1]
                progressive_time = progressive_time[:-1]
                mean_n_stage = mean_n_stage[:-1]

            all_times.append(progressive_time)
            all_populations.append(populations_stage)
            all_mean_n.append(mean_n_stage)
            
            time_changes.append(progressive_time[-1])

            current_time_offset += t[-1]

        times = np.concatenate(all_times) * 1e-3  # Convert to ms
        populations = np.concatenate(all_populations, axis=1)
        mean_n = np.concatenate(all_mean_n)

    # --- SINGLE-STAGE CASE HANDLING ---
    else:
        res = results
        populations = np.zeros((n_internal, len(times)))
        mean_n = []

        n_op = tensor(qeye(n_internal), num(n_motional))

        for i, state in enumerate(res.states):
            for j in range(n_internal):
                Pj_op = tensor(
                    basis(n_internal, j) * basis(n_internal, j).dag(), 
                    qeye(n_motional)
                )
                populations[j, i] = expect(Pj_op, state)

            mean_n.append(expect(n_op, state))

        times = times * 1e-3  # Convert to ms

    # --- PLOTTING LOGIC: ONLY POPULATION HEATMAP ---
    if only_pop:
        fig, axs = plt.subplots(figsize=(15, 5))

        im = axs.imshow(
            populations, aspect='auto', origin='lower', 
            extent=[times[0], times[-1], -0.5, n_internal - 0.5], 
            cmap='viridis'
        )
        
        axs.set_xlabel(r'Time ($ms$)', fontsize=25)
        axs.set_ylabel('Internal level index', fontsize=25)
        axs.set_xlim(0, 90)
        
        xticks = np.arange(times[0], times[-1] + 10, 10)
        axs.set_xticks(xticks)
        axs.set_xticklabels([f"{int(x)}" for x in xticks], fontsize=20)

        for t_mark in time_changes[:-1]:
            axs.axvline(
                t_mark * 1e-3, color="red", linestyle="--", 
                linewidth=1.5, alpha=0.7
            )

        secax = axs.secondary_yaxis('right', functions=(lambda y: y, lambda y: y))
        special_positions = [0, 1, int(n_internal / 2)]
        special_labels = ["LM", "PU", "PL"]
        secax.set_yticks(special_positions)
        secax.set_yticklabels(special_labels, fontsize=18)
        secax.minorticks_off()
        secax.tick_params(axis='y', which='both', length=10, direction='inout')

        if title:
            axs.set_title(title, fontsize=28)
        else:
            axs.set_title('Population evolution (internal levels)', fontsize=28)

        cbar = fig.colorbar(im, ax=axs, label="Population", pad=0.12)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label("Population", fontsize=25)

        axs.tick_params(axis='both', pad=10, labelsize=20)

        plt.tight_layout()
        if savetext:
            save_figure_in_images(fig, filename=savetext)
        plt.show()

    # --- PLOTTING LOGIC: HEATMAP + PHONON DYNAMICS ---
    else:
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))

        im = axs[0].imshow(
            populations, aspect='auto', origin='lower', 
            extent=[times[0], times[-1], -0.5, n_internal - 0.5], 
            cmap='viridis'
        )
        axs[0].set_xlabel(r'Time ($ms$)', fontsize=25)
        axs[0].set_ylabel('Internal level index', fontsize=25)
        
        xticks = np.linspace(times[0], times[-1], 6)
        axs[0].set_xticks(xticks)
        axs[0].set_xticklabels([f"{int(x)}" for x in xticks], fontsize=20)

        secax = axs[0].secondary_yaxis('right', functions=(lambda y: y, lambda y: y))
        special_positions = [0, 1, int(n_internal / 2)]
        special_labels = ["LM", "PU", "PL"]
        secax.set_yticks(special_positions)
        secax.set_yticklabels(special_labels, fontsize=18)
        secax.minorticks_off()
        secax.tick_params(axis='y', which='both', length=10, direction='inout')

        if title:
            axs[0].set_title(title, fontsize=28)
        else:
            axs[0].set_title('Population evolution (internal levels)', fontsize=28)

        cbar = fig.colorbar(im, ax=axs[0], label="Population", pad=0.12)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label("Population", fontsize=25)

        axs[1].plot(times, mean_n, color='darkorange', lw=2)
        axs[1].set_xlabel('Time (ms)', fontsize=22)
        axs[1].set_ylabel(r'$\langle n \rangle$ (Mean phonon number)', fontsize=22)
        
        if title:
            axs[1].set_title(title, fontsize=24)
        else:
            axs[1].set_title('Motional excitation', fontsize=24)
        axs[1].grid(True)

        for ax in axs:
            ax.tick_params(axis='both', pad=10, labelsize=20)

        plt.tight_layout()
        if savetext:
            save_figure_in_images(fig, filename=savetext)
        plt.show()