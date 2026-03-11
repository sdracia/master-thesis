"""
Visualization Module for Rapid Adiabatic Passage (RAP) Pulse Dynamics.

This module provides tools to plot the temporal evolution of a molecule-ion system 
during a RAP sequence. It specifically tracks internal electronic populations 
and motional phonon excitations (mean phonon number) over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import Result, tensor, qeye, num, basis, expect
from typing import Optional, Union, List
from saving import save_figure_in_images


def RAP_plot_internal_and_motional_dynamics(
    result: Result,
    times: np.ndarray,
    n_motional: int,
    j_val: Union[int, float],
    dm: int,
    result_rf: Optional[Result] = None,
    j_plot: Optional[Union[int, List[int]]] = None,
    savetext: str = "rap_pulse"
) -> float:
    """
    Plots the population dynamics of internal states and motional excitation.

    Parameters
    ----------
    result : qutip.Result
        Result object containing evolved quantum states from the RAP simulation.
    times : np.ndarray
        Array of time points (in microseconds or equivalent simulation units).
    n_motional : int
        Dimension of the motional Hilbert space (Fock space cutoff).
    j_val : float or int
        Total angular momentum quantum number of the manifold being simulated.
    dm : int
        Change in magnetic quantum number characterizing the transition.
    result_rf : qutip.Result, optional
        Secondary simulation result (e.g., Rotating Frame or Rabi Flop) for 
        comparison data processing.
    j_plot : int or list of int, optional
        Manifold index (or list of indices) to enable plotting. If None, 
        plotting is always performed.
    savetext : str, optional
        Prefix for the saved figure filename. Default is "rap_pulse".

    Returns
    -------
    last_up_pop : float
        The population of the excited internal state |1> at the final time step.
    """

    # Determine whether to generate the plot based on the manifold index
    if j_plot is None:
        do_plot = True
    elif isinstance(j_plot, int):
        do_plot = (j_val == j_plot)
    else:
        do_plot = (j_val in j_plot)

    # Initialize population matrix (2 states: ground and excited)
    populations = np.zeros((2, len(times)))
    mean_n = []
    
    # Define the phonon number operator: I_internal ⊗ n_motion
    n_op = tensor(qeye(2), num(n_motional))

    # Calculate expectation values for the primary simulation
    for i, state in enumerate(result.states):
        for j in range(2):
            # Projector for the j-th internal state: |j><j| ⊗ I_motion
            Pj_op = tensor(
                basis(2, j) * basis(2, j).dag(), 
                qeye(n_motional)
            )
            populations[j, i] = expect(Pj_op, state)

        mean_n.append(expect(n_op, state))

    # Calculate comparison data if result_rf is provided
    if result_rf is not None:
        populations_rf = np.zeros((2, len(times)))
        mean_n_rf = []

        for i, state in enumerate(result_rf.states):
            for j in range(2):
                Pj_op_rf = tensor(
                    basis(2, j) * basis(2, j).dag(), 
                    qeye(n_motional)
                )
                populations_rf[j, i] = expect(Pj_op_rf, state)

            mean_n_rf.append(expect(n_op, state))

    # --- Plotting Section ---
    if do_plot:
        fig, axs = plt.subplots(1, 2, figsize=(16, 4))

        # Subplot 0: Internal State Populations
        # Scaling time by 1e-3 to convert to ms
        axs[0].plot(
            times * 1e-3, populations[0], 
            label=r'$|0\rangle$', color='tab:blue', linestyle='-'
        )
        axs[0].plot(
            times * 1e-3, populations[1], 
            label=r'$|1\rangle$', color='tab:orange', linestyle='-'
        )

        axs[0].set_xlabel(r'Time (ms)', fontsize=20)
        axs[0].set_ylabel('Population', fontsize=20)
        axs[0].set_title(
            rf'RAP pulse dynamics, J = {j_val}', 
            fontsize=25
        )
        axs[0].legend(fontsize=16, loc='center left')
        axs[0].grid(True)
        axs[0].tick_params(axis='both', labelsize=20)

        # Subplot 1: Mean Phonon Number (Motional Excitation)
        axs[1].plot(
            times * 1e-3, mean_n, 
            lw=2, color="tab:orange"
        )
        axs[1].set_xlabel(r'Time (ms)', fontsize=20)
        axs[1].set_ylabel(r'$\langle n \rangle$ (Mean phonon number)', fontsize=20)
        axs[1].set_title('Motional excitation', fontsize=20)
        axs[1].grid(True)
        axs[1].tick_params(axis='both', labelsize=20)

        plt.tight_layout()

        # Save the figure with a specific name for the manifold
        filename = f"{savetext}_j{j_val}.svg"
        save_figure_in_images(fig, filename)

        plt.show()

    # Extract the final transfer efficiency (population in |1>)
    last_up_pop = populations[1][-1]

    return last_up_pop