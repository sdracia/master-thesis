"""
Module to simulate and visualize the dynamics of a three-level quantum system
with optional adiabatic elimination to a two-level approximation.

This module provides:
- Time evolution of a three-level system under coherent drive.
- Comparison with the effective two-level approximation.
- Plotting functions with optional zoom-in for small populations.
"""

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

matplotlib.pyplot.style.use(['science', 'notebook', 'ieee'])
matplotlib.pyplot.rcParams['font.family'] = 'Times New Roman'

from saving import save_figure_in_images



def dCdt(t, C, Omega, Delta1, Delta2):
    """
    Compute the differential equations for the three-level system state vector.

    Parameters
    ----------
    t : float
        Current time.
    C : list of complex
        State vector [C1, C2, C3].
    Omega : float
        Rabi frequency for the transitions.
    Delta1 : float
        Detuning of first transition.
    Delta2 : float
        Detuning of second transition.

    Returns
    -------
    list of complex
        Time derivatives [dC1/dt, dC2/dt, dC3/dt].
    """
    C1, C2, C3 = C
    # Equations of motion for the three-level system
    dC1 = -1j*Delta1*C1 - 1j * Omega / 2 * C2
    dC2 = -1j * Omega / 2 * C1 - 1j * Omega / 2 * C3
    dC3 = -1j * Omega / 2 * C2 - 1j*Delta2*C3
    return [dC1, dC2, dC3]



def adiabatic_elimination(Omega, Delta1, Delta2, filename=None, zoom=False):
    """
    Simulate the three-level system dynamics and compare with two-level approximation.

    Parameters
    ----------
    Omega : float
        Rabi frequency for the transitions.
    Delta1 : float
        Detuning of first transition.
    Delta2 : float
        Detuning of second transition.
    filename : str, optional
        File name to save the resulting plot. If None, a default name is used.
    zoom : bool, optional
        If True, produce a zoomed-in plot highlighting small population dynamics.

    Returns
    -------
    None
    """
    # Initial state: system starts in |2> state
    C0 = [0.0 + 0j, 1.0 + 0j, 0.0 + 0j]

    # Time span for integration
    t_span = (0, 1e6 * Omega * 80)
    t_eval = np.linspace(*t_span, 1000)

    # Solve the time evolution
    sol = solve_ivp(dCdt, t_span, C0, t_eval=t_eval, args=(Omega, Delta1, Delta2))

    # Populations |Cj(t)|^2
    pop1 = np.abs(sol.y[0])**2
    pop2 = np.abs(sol.y[1])**2
    pop3 = np.abs(sol.y[2])**2

    # Two-level approximation
    pop3_ideal = Omega**2 / (Omega**2 + Delta1**2) * np.sin(Omega * t_eval / 2)**2

    # Plot results
    plot_adibatic_elimination(sol, pop1, pop2, pop3, pop3_ideal, t_eval, filename, zoom)



def plot_adibatic_elimination(sol, pop1, pop2, pop3, pop3_ideal, t_eval, filename=None, zoom=False):
    """
    Plot the population dynamics of the three-level system with optional zoom.

    Parameters
    ----------
    sol : OdeResult
        Solution object from solve_ivp.
    pop1 : np.ndarray
        Population of state |C1>.
    pop2 : np.ndarray
        Population of state |C2>.
    pop3 : np.ndarray
        Population of state |C3>.
    pop3_ideal : np.ndarray
        Population of |C3> in the two-level approximation.
    t_eval : np.ndarray
        Array of times used in the integration.
    filename : str, optional
        File name to save the figure. Default uses 'adiabatic_elimination.svg'.
    zoom : bool, optional
        If True, produce a zoomed-in plot to highlight small populations.

    Returns
    -------
    None
    """
    if not zoom:
        # Standard full plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sol.t, pop1, label='$|C_b(t)|^2$ - (3-level system)', color="blue", linestyle="-", linewidth=4)
        ax.plot(sol.t, pop2, label='$|C_a(t)|^2$ - (3-level system)', color="orange", linestyle="-", linewidth=4)
        ax.plot(sol.t, pop3, label='$|C_c(t)|^2$ - (3-level system)', color="green", linestyle="-", linewidth=4)
        ax.plot(t_eval, pop3_ideal, '--', label='$|C_b(t)|^2$ - (2-level approx.)', color='black')

        ax.set_xlabel('Time (μs)', fontsize=25)
        ax.set_ylabel('Population', fontsize=25)
        ax.set_title('Dynamics of three-Level system with two-level approximation', fontsize=28)
        ax.legend(fontsize=21, frameon=True, loc='upper right')
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=25)
        fig.tight_layout()
    
    else:
        # Zoomed-in plot
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6),
            gridspec_kw={'height_ratios': [1,1]}
        )

        # Upper region: near 1 population
        ax1.plot(sol.t, pop2, label='$|C_a(t)|^2$ - (3-level system)', color='orange', linestyle="-", linewidth=4)
        ax1.set_ylim(0.99, 1)
        ax1.set_xlim(0, 15000)

        # Lower region: near 0 population
        ax2.plot(sol.t, pop1, label='$|C_b(t)|^2$ - (3-level system)', color='blue', linestyle="-", linewidth=4)
        ax2.plot(sol.t, pop3, label='$|C_c(t)|^2$ - (3-level system)', color='green', linestyle="-", linewidth=4)
        ax2.plot(t_eval, pop3_ideal, '--', label='$|C_b(t)|^2$ - (2-level approx.)', color='black')
        ax2.set_ylim(0, 0.01)
        ax2.set_xlim(0, 15000)

        # Adjust spacing and hide common spines
        plt.subplots_adjust(hspace=0.2)
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.tick_params(labeltop=False, labelsize=25)

        # Draw break marks
        d = 0.015  # size of break marks
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

        # Grid, labels, and unified legend
        ax1.grid(True)
        ax2.grid(True)
        fig.supxlabel("Time (μs)", fontsize=25)
        fig.supylabel("Population", fontsize=25).set_x(-0.01)
        fig.suptitle('Dynamics of three-Level system with two-level approximation', fontsize=28)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        fig.legend(handles, labels, frameon=True, loc='center')
        plt.tick_params(axis='both', which='major', labelsize=25)

    # Save figure
    if filename is not None:
        save_figure_in_images(fig, filename)
    else:
        save_figure_in_images(fig, "adiabatic_elimination.svg")

    plt.show()