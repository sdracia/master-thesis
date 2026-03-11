# File: exp_imperfections.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Module for analyzing molecular transition variations and applying experimental imperfections.

This module provides functions to:
- Apply noise to values for simulating experimental imperfections.
- Add false positive excitations to molecular spectra.
- Compute exact or approximate transition energies for different types.
- Plot relative variations of transitions over parameters (g_j or c_ij).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, physical_constants

from saving import save_figure_in_images
from molecules.molecule import CaOH


# Nuclear magneton and proton g factor for calculations
mu_N = physical_constants["nuclear magneton"][0]
gI = physical_constants["proton g factor"][0]


def apply_noise(value, noise_type, noise_level, seed=None):
    """
    Apply noise to a value to simulate experimental imperfections.

    Parameters
    ----------
    value : float
        Original numerical value to which noise will be applied.
    noise_type : str
        Type of noise: 'rel_uniform', 'rel_gaussian', 'abs_uniform', 'abs_gaussian', 'value'.
    noise_level : float
        Percentage or absolute magnitude of error.
    seed : int, optional
        Random seed for reproducibility. If None, noise is random each call. Default is None.

    Returns
    -------
    float
        Value with noise applied.
    """
    rng = np.random.default_rng(seed)  # Initialize random generator
    
    # Relative or absolute noise
    if noise_type == "rel_uniform":
        error = noise_level * np.abs(value) * rng.uniform(-1, 1)
    elif noise_type == "rel_gaussian" or noise_type == "rel_normal":
        error = noise_level * np.abs(value) * rng.normal(0, 1)
    elif noise_type == "abs_uniform":
        error = noise_level * rng.uniform(-1, 1)
    elif noise_type == "abs_gaussian" or noise_type == "abs_normal":
        error = noise_level * rng.normal(0, 1)
    elif noise_type == "value":
        error = noise_level
    else:
        error = 0  # No noise if unknown type

    return value + error


def false_positive_excitation(frequencies: np.ndarray,
                              exc_probs: np.ndarray,
                              false_positive_rate: float = 0.0,
                              type_false_positive: str = "uniform") -> np.ndarray:
    """
    Adds false positive excitations to a molecular spectrum.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies of the spectrum.
    exc_probs : np.ndarray
        Array of excitation probabilities corresponding to each frequency.
    false_positive_rate : float
        Magnitude of the false positive contribution to add.
    type_false_positive : str, optional
        Type of false positive to add: 'uniform' or 'gaussian'. Default is 'uniform'.

    Returns
    -------
    np.ndarray
        Excitation probabilities including the false positive contributions.
    """
    if false_positive_rate == 0.0:
        return exc_probs

    # Add false positives depending on chosen type
    if type_false_positive == "uniform":
        false_positive_excitation = np.random.uniform(0, false_positive_rate, size=len(frequencies))
    elif type_false_positive == "gaussian":
        false_positive_excitation = np.random.normal(0, false_positive_rate, size=len(frequencies))
    else:
        raise ValueError(f"Unknown type of false positive excitation: {type_false_positive}")

    return exc_probs + false_positive_excitation


def compute_transition_type(mo1, gj, cij, transition_type="signature"):
    """
    Computes transition energies for a molecule for a given transition type.

    Parameters
    ----------
    mo1 : Molecule
        Molecule object containing the transition DataFrame and properties.
    gj : float
        g_j value of the molecule.
    cij : float
        Coupling constant c_ij in kHz.
    transition_type : str, optional
        Type of transition to compute. Options are:
        'signature', 'penultimate_upper', 'penultimate_lower', 'sub_manifold_splitting'. Default is 'signature'.

    Returns
    -------
    np.ndarray
        Array of computed transition energies for each J manifold.
    """
    if transition_type == "signature":
        transition_exact = np.abs(np.array([mo1.transition_df.loc[mo1.transition_df["j"]==j].iloc[0]["energy_diff"]
                                            for j in range(1, mo1.j_max+1)]))
    elif transition_type == "penultimate_upper":
        transition_exact = np.abs(np.array([mo1.transition_df.loc[mo1.transition_df["j"]==j].iloc[1]["energy_diff"]
                                            for j in range(1, mo1.j_max+1)]))
    elif transition_type == "penultimate_lower":
        transition_exact = np.abs(np.array([mo1.transition_df.loc[mo1.transition_df["j"]==j].iloc[(2*j+1)*2 - 2]["energy_diff"]
                                            for j in range(1, mo1.j_max+1)]))
    elif transition_type == "sub_manifold_splitting":
        transition_exact = []
        for i, j in enumerate(range(1, mo1.j_max + 1)):
            # Compute splitting based on coupling and g-factor difference
            x = 1/2 * np.sqrt(cij**2 * ((j + 1/2)**2) + (- mo1.cb_khz * (gj - gI))**2)
            transition_exact.append(2 * x)
        transition_exact = np.array(transition_exact)
    else:
        print("Invalid type. Choose between signature, penultimate_upper, penultimate_lower, and sub_manifold_splitting.")

    return transition_exact


def plot_variation_transition(j_max, relative_matrix, parameter, range_param, contours=None, filename="STvsGJcaoh.svg"):
    """
    Plot the relative variation of molecular transitions across J manifolds for a varying parameter.

    Parameters
    ----------
    j_max : int
        Maximum J manifold considered.
    relative_matrix : np.ndarray
        Matrix of relative errors for the transitions to be plotted.
    parameter : str
        Parameter being varied; must be 'gj' or 'cij'.
    range_param : np.ndarray
        Array of parameter values over which the variation is computed.
    contours : list of tuples, optional
        List of (level, color) tuples for overlaying contour lines on the heatmap. Default is None.
    filename : str, optional
        Name of the file where the figure will be saved. Defaults to 'STvsGJcaoh.svg'. Default is "STvsGJcaoh.svg".

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 11))  # Single axis

    X, Y = np.meshgrid(np.arange(1, j_max + 1), range_param)
    Z = np.abs(np.array(relative_matrix))

    # Display heatmap
    im = ax.imshow(Z, aspect='auto', origin='lower',
                   extent=[1, j_max + 1, range_param[0], range_param[-1]],
                   cmap='coolwarm')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Percentage variation', fontsize=25)
    cbar.ax.yaxis.set_tick_params(labelsize=25)

    # Optional contours
    if contours is not None:
        for level, color in contours:
            contour = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=3)
            ax.clabel(contour, fmt=f"{level}", colors=color, fontsize=25)

    # Labels and title
    ax.set_xlabel('$J$', fontsize=25)
    if parameter == "gj":
        ax.set_ylabel('$g_{j}$', fontsize=25)
        ax.set_title('Signature transition relative variation over $g_{j}$', fontsize=28)
    elif parameter == "cij":
        ax.set_ylabel('$c_{ij}$', fontsize=25)
        ax.set_title('Signature transition relative variation over $c_{ij}$', fontsize=28)
    else:
        print("Invalid parameter. Choose between cij and gj.")

    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()

    # Save figure
    save_figure_in_images(fig, filename if filename else "STvsGJcaoh.svg")
    plt.show()


def transition_relative_variation(b_field_gauss, j_max,
                                  range_param, contours=None,
                                  parameter="gj", transition_type="signature",
                                  filename="STvsGJcaoh.svg"):
    """
    Compute and plot the relative variation of molecular transition energies across a parameter sweep.

    Parameters
    ----------
    b_field_gauss : float
        Magnetic field applied to the molecule in Gauss.
    j_max : int
        Maximum J manifold considered in the computation.
    range_param : np.ndarray
        Array of parameter values over which to sweep.
    contours : list of tuples, optional
        List of (level, color) tuples to overlay contour lines on the plot. Default is None.
    parameter : str, optional
        Parameter to vary; must be 'gj' or 'cij'. Default is 'gj'.
    transition_type : str, optional
        Type of transition to consider; e.g., 'signature', 'penultimate_upper', 'penultimate_lower', 'sub_manifold_splitting'. Default is 'signature'.
    filename : str, optional
        Name of the file to save the resulting figure. Default is "STvsGJcaoh.svg".

    Returns
    -------
    None
    """
    cij = CaOH.cij_khz
    gj = CaOH.gj

    mo1 = CaOH.create_molecule_data(b_field_gauss=b_field_gauss, j_max=j_max)
    transition_exact = compute_transition_type(mo1, gj, cij, transition_type)

    transition_matrix = []
    difference_matrix = []
    relative_matrix = []

    # Sweep over the parameter values
    for par in range_param:
        if parameter == "gj":
            CaOH.gj = par
        elif parameter == "cij":
            CaOH.cij_khz = par
        else:
            print("Invalid parameter. Choose between cij and gj.")

        mo1 = CaOH.create_molecule_data(b_field_gauss=b_field_gauss, j_max=j_max)
        transition_noised = compute_transition_type(mo1, gj, cij, transition_type)

        diff = transition_noised - transition_exact
        rel_err = diff / transition_exact

        difference_matrix.append(diff)
        relative_matrix.append(rel_err)
        transition_matrix.append(transition_noised)

    transition_matrix = np.array(transition_matrix)

    # Plot results
    plot_variation_transition(j_max, relative_matrix, parameter, range_param, contours, filename)

    # Reset parameters to default
    CaOH.cij_khz = 1.49
    CaOH.gj = -0.036