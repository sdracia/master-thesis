# File: state_preparation.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Molecular Pumping and Adiabatic Passage (RAP) Simulation Pipeline.

This module provides a comprehensive pipeline to analyze molecular population 
distributions after pumping stages and Rapid Adiabatic Passage (RAP) sequences.
It includes utilities for:
- Automatic directory discovery in the project tree.
- Sigmoid fitting of pumping efficiencies across rotational manifolds.
- Population redistribution modeling for CaH and CaOH species.
- High-quality visualization for thesis-level reporting.
"""

import re
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from qutip import tensor, basis, qeye, expect
from typing import Dict, List, Tuple, Any, Optional, Union

from QLS.state_dist import States
from saving import save_figure_in_images


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def find_data_dir(folder_name: str, start_path: Path = Path.cwd()) -> Path:
    """
    Searches for a specific folder by climbing up the filesystem tree.

    Parameters
    ----------
    folder_name : str
        Name of the directory to find.
    start_path : Path, optional
        The path to start the search from. Defaults to current working directory.

    Returns
    -------
    Path
        The absolute path to the found directory.

    Raises
    ------
    FileNotFoundError
        If the directory is not found after reaching the filesystem root.
    """
    current = start_path.resolve()

    while True:
        candidate = current / folder_name
        if candidate.exists() and candidate.is_dir():
            return candidate

        if current.parent == current:
            raise FileNotFoundError(
                f"Directory '{folder_name}' not found climbing up from {start_path}"
            )
        current = current.parent


def _load_common(
    molecule: Any, 
    temperature: float, 
    data_dir: Path
) -> Tuple[States, np.ndarray, List[Path]]:
    """
    Common loading logic for CaH and CaOH species.

    Parameters
    ----------
    molecule : Molecule
        The molecule object containing transition data.
    temperature : float
        Rotational temperature for thermal distribution.
    data_dir : Path
        Directory where pickle data files are stored.

    Returns
    -------
    tuple
        (states object, J-distribution array, sorted list of data file paths).
    """
    states = States(molecule, temperature)
    j_dist = states.j_distribution()

    # Match files like J1_..., J10_..., etc.
    pattern = re.compile(r"J(\d+)_")

    data_files = sorted(
        [f for f in data_dir.glob("*.pkl") if pattern.match(f.name)],
        key=lambda f: int(pattern.match(f.name).group(1))
    )

    return states, j_dist, data_files


# ==========================================================
# CaH PIPELINE
# ==========================

def _run_cah(
    molecule: Any, 
    temperature: float, 
    j_max: int, 
    plot: bool = False, 
    rap_flag: bool = True, 
    savetext: str = "cah_after_pumping_rap_pop.svg"
) -> Dict[str, Any]:
    """
    Internal execution pipeline for CaH molecule analysis.
    """
    data_dir = find_data_dir("pumping_RAP_data_cah")

    states, j_dist, data_files = _load_common(molecule, temperature, data_dir)

    # Calculate fits for pumping efficiencies
    pop_fit_0, pop_fit_1, pop_fit_2jp1 = pumping_manifolds_cah(
        data_files, 
        plot=plot
    )

    # Load pre-calculated RAP signatures
    data_cah = joblib.load(data_dir / "rap_signature_cah.pkl")
    pop_vals_cah = data_cah["pop_vals"]

    # Calculate final population distribution
    (
        pop0_before, pop1_before, pop2jp1_before,
        pop0_after, pop1_after, pop2jp1_after
    ) = after_pumping_rap_pop_cah(
        molecule=molecule,
        state_dist=states,
        j_max=j_max,
        pop_fit_0=pop_fit_0,
        pop_fit_1=pop_fit_1,
        pop_fit_2jp1=pop_fit_2jp1,
        rap_sign_cah=pop_vals_cah,
        plot=plot,
        rap_flag=rap_flag,
        savetext=savetext
    )

    pop_fit = [pop_fit_0, pop_fit_1, pop_fit_2jp1]

    return {
        "states": states,
        "j_dist": j_dist,
        "pop_fit": pop_fit,
        "before": (pop0_before, pop1_before, pop2jp1_before),
        "after": (pop0_after, pop1_after, pop2jp1_after),
    }


# ==========================================================
# CaOH PIPELINE
# ==========================================================

def _run_caoh(
    molecule: Any, 
    temperature: float, 
    j_max: int, 
    plot: bool = False, 
    rap_flag: bool = True, 
    savetext: str = "caoh_after_pumping_rap_pop.svg"
) -> Dict[str, Any]:
    """
    Internal execution pipeline for CaOH molecule analysis.
    """
    data_dir = find_data_dir("pumping_RAP_data_caoh")

    states, j_dist, data_files = _load_common(molecule, temperature, data_dir)

    pop_fit_0, pop_fit_1, pop_fit_2jp1 = pumping_manifolds(
        data_files, 
        plot=plot
    )

    # Load specific RAP signatures for the different transition sets
    data_high = joblib.load(data_dir / "rap_signature_high.pkl")
    data_middle = joblib.load(data_dir / "rap_signature_middle.pkl")
    data_low = joblib.load(data_dir / "rap_signature_low.pkl")
    data_LL = joblib.load(data_dir / "rap_signature_LL.pkl")

    # Extract signature values
    pop_vals_high = data_high["pop_vals"]
    pop_vals_middle = data_middle["pop_vals"]
    pop_vals_low = data_low["pop_vals"]
    pop_vals_LL = data_LL["pop_vals"]

    (
        pop0_before, pop1_before, pop2jp1_before,
        pop0_after, pop1_after, pop2jp1_after
    ) = after_pumping_rap_pop(
        molecule=molecule,
        state_dist=states,
        j_max=j_max,
        pop_fit_0=pop_fit_0,
        pop_fit_1=pop_fit_1,
        pop_fit_2jp1=pop_fit_2jp1,
        rap_sign_low=pop_vals_low,
        rap_sign_middle=pop_vals_middle,
        rap_sign_high=pop_vals_high,
        rap_sign_LL=pop_vals_LL,
        plot=plot,
        rap_flag=rap_flag,
        savetext=savetext
    )

    pop_fit = [pop_fit_0, pop_fit_1, pop_fit_2jp1]

    return {
        "states": states,
        "j_dist": j_dist,
        "pop_fit": pop_fit,
        "before": (pop0_before, pop1_before, pop2jp1_before),
        "after": (pop0_after, pop1_after, pop2jp1_after),
    }


def run_pumping_pipeline(
    molecule_type: str,
    molecule: Any,
    temperature: float,
    j_max: int,
    plot: bool = False,
    rap_flag: bool = True
) -> Dict[str, Any]:
    """
    Main entry point wrapper to execute the pipeline for CaH or CaOH.

    Parameters
    ----------
    molecule_type : str
        The molecule species ('CaH' or 'CaOH').
    molecule : Molecule
        Object containing the molecular structure/Hamiltonian.
    temperature : float
        Rotational temperature in Kelvin.
    j_max : int
        Maximum rotational level to consider.
    plot : bool, optional
        Whether to generate visualization plots.
    rap_flag : bool, optional
        If True, applies RAP population transfer signatures.

    Returns
    -------
    dict
        Dictionary containing states, distributions, fits, and before/after populations.
    """
    molecule_type = molecule_type.lower()

    if molecule_type == "cah":
        return _run_cah(
            molecule, temperature, j_max, plot, rap_flag, 
            savetext="cah_after_pumping_rap_pop.svg"
        )
    elif molecule_type == "caoh":
        return _run_caoh(
            molecule, temperature, j_max, plot, rap_flag, 
            savetext="caoh_after_pumping_rap_pop.svg"
        )
    else:
        raise ValueError(
            f"Unknown molecule_type '{molecule_type}'. Supported: 'CaH', 'CaOH'."
        )


# ==========================================================
# CORE PROCESSING & FITTING LOGIC
# ==========================================================

def fit_populations(
    popj_dict: Dict[int, float], 
    type: str, 
    no_plot: bool = True, 
    ax: Optional[plt.Axes] = None
) -> np.ndarray:
    """
    Fits internal state populations across J manifolds using sigmoid models.

    Parameters
    ----------
    popj_dict : dict
        Mapping of J-manifold index to observed population.
    type : str
        Fitting model selector: 'popj_1' or 'popj_2jp1'.
    no_plot : bool, optional
        If False, plots the fitting results on the provided axis.
    ax : plt.Axes, optional
        The axis to plot on if no_plot is False.

    Returns
    -------
    np.ndarray
        Interpolated population values from J=1 to J=50.
    """
    j_vals = np.array(sorted(popj_dict.keys()))
    pop_vals = np.array([popj_dict[j] for j in j_vals])

    # Sigmoid definitions for different target states
    def sigmoid_1(J, A, B, C):
        return 0.5 / (1 + A * np.exp(-B * (J - C)))
    
    def sigmoid_2jp1(J, A, B, C):
        return 0.5 - 0.5 / (1 + A * np.exp(-B * (J - C)))

    if type == "popj_1":
        popt, _ = curve_fit(sigmoid_1, j_vals, pop_vals)
    elif type == "popj_2jp1":
        popt, _ = curve_fit(sigmoid_2jp1, j_vals, pop_vals)
    else:
        print("Invalid type. Choose 'popj_1' or 'popj_2jp1'.")

    A_fit, B_fit, C_fit = popt
    j_fit = np.arange(1, 51)

    if type == "popj_1":
        pop_fit = sigmoid_1(j_fit, A_fit, B_fit, C_fit)
        color, label = '#4B0082', "PU state"
    elif type == "popj_2jp1":
        pop_fit = sigmoid_2jp1(j_fit, A_fit, B_fit, C_fit)
        color, label = '#DAA520', "PL state"
    else:
        print("Invalid type. Choose 'popj_1' or 'popj_2jp1'.")

    if not no_plot:
        ax.scatter(j_vals, pop_vals, label=label, color=color)
        ax.plot(j_fit, pop_fit, label="fit " + label, color=color, linestyle='-')

    return pop_fit


def pumping_manifolds(
    data_files: List[Path], 
    plot: bool = True, 
    savetext: str = "caoh_after_pumping.svg"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes QuTiP density matrices to extract manifold populations for CaOH.
    """
    popj_0_dict = {}
    popj_1_dict = {}
    popj_2jp1_dict = {}

    for file_path in data_files:
        data = joblib.load(file_path)
        j_val = data["j_val"]
        rho_final = data["rho_final"]
        args = data["args"]
        n_motional = args["terms"][0]["n_motional"]
        n_internal = args["terms"][0]["n_internal"]

        populations = np.zeros(n_internal)
        
        # Calculate expectation values for each internal level
        for j in range(n_internal):
            Pj_op = tensor(basis(n_internal, j) * basis(n_internal, j).dag(), qeye(n_motional))
            populations[j] = expect(Pj_op, rho_final)

        popj_0 = populations[0]
        popj_1 = populations[1]

        # Empirical correction for high J manifolds
        if j_val >= 11:
            popj_0 = popj_0 - 0.1 + 1 / (2 * (2 * j_val + 1)) 
            popj_1 = popj_1 + 0.1 - 1 / (2 * (2 * j_val + 1)) 

        index = min(2 * j_val + 1, int(n_internal / 2))
        popj_2jp1 = populations[index]

        popj_0_dict[j_val], popj_1_dict[j_val], popj_2jp1_dict[j_val] = popj_0, popj_1, popj_2jp1

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        pop_fit_1 = fit_populations(popj_1_dict, "popj_1", no_plot=False, ax=ax)
        pop_fit_2jp1 = fit_populations(popj_2jp1_dict, "popj_2jp1", no_plot=False, ax=ax)
        pop_fit_0 = 1 - pop_fit_1 - pop_fit_2jp1

        j_vals_0 = np.array(sorted(popj_0_dict.keys()))
        pop_vals_0 = np.array([popj_0_dict[j] for j in j_vals_0])
        j_fit = np.arange(1, 51)

        ax.scatter(j_vals_0, pop_vals_0, label="LM (target) state", color='#800020')
        ax.plot(j_fit, pop_fit_0, label="fit LM state", color='#800020', linestyle='-')
        ax.set_xlabel(r'Manifold $J$', fontsize=25)
        ax.set_ylabel('Population', fontsize=25)
        ax.set_title('Population after pumping stage', fontsize=28)
        ax.grid(True)
        ax.legend(fontsize=21, frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        save_figure_in_images(fig, savetext)
        plt.show()
    else:
        pop_fit_1 = fit_populations(popj_1_dict, "popj_1", no_plot = True)
        pop_fit_2jp1 = fit_populations(popj_2jp1_dict, "popj_2jp1", no_plot = True)
        pop_fit_0 = 1 - pop_fit_1 - pop_fit_2jp1


        j_vals_0 = np.array(sorted(popj_0_dict.keys()))
        pop_vals_0 = np.array([popj_0_dict[j] for j in j_vals_0])
        j_fit = np.arange(1, 51)


    return pop_fit_0, pop_fit_1, pop_fit_2jp1


def after_pumping_rap_pop(
    molecule: Any, 
    state_dist: States, 
    j_max: int, 
    pop_fit_0: np.ndarray, 
    pop_fit_1: np.ndarray, 
    pop_fit_2jp1: np.ndarray,
    rap_sign_low: np.ndarray, 
    rap_sign_middle: np.ndarray, 
    rap_sign_high: np.ndarray, 
    rap_sign_LL: np.ndarray, 
    plot: bool = True, 
    rap_flag: bool = True,
    savetext: str = "caoh_after_pumping_rap_pop.svg"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies Rapid Adiabatic Passage (RAP) signatures to the pumped populations.
    """
    thermal_distribution = state_dist.j_distribution()
    molecule.state_df.loc[molecule.state_df.index[2:], "state_dist"] = 0.0

    if not np.allclose(pop_fit_0 + pop_fit_1 + pop_fit_2jp1, 1.0, atol=1e-6):
        raise ValueError("Population is not normalized to 1")
    
    modified_pop0, modified_pop1, modified_pop2 = [], [], []

    for j_val in range(1, j_max + 1):
        tot_pop_in_j = thermal_distribution[j_val]

        # Initial populations from pumping fits
        pop0 = pop_fit_0[j_val - 1] * tot_pop_in_j
        pop1 = pop_fit_1[j_val - 1] * tot_pop_in_j
        pop2 = pop_fit_2jp1[j_val - 1] * tot_pop_in_j

        if rap_flag:
            # First RAP
            transfer = rap_sign_low[j_val] * pop1
            pop0 = pop0 + transfer
            pop1 = pop1 - transfer

            # Second RAP
            transfer = rap_sign_middle[j_val] * pop1
            pop0 = pop0 + transfer
            pop1 = pop1 - transfer

            # Third RAP
            transfer = rap_sign_high[j_val] * pop1
            pop0 = pop0 + transfer
            pop1 = pop1 - transfer

            # Fourth RAP
            transfer = rap_sign_LL[j_val] * pop2
            pop0 = pop0 + transfer
            pop2 = pop2 - transfer

        modified_pop0.append(pop0 / tot_pop_in_j)
        modified_pop1.append(pop1 / tot_pop_in_j)
        modified_pop2.append(pop2 / tot_pop_in_j)


        states_in_j = molecule.state_df.loc[molecule.state_df["j"] == j_val].copy()

        # Update main molecule dataframe
        indices_in_j = molecule.state_df.index[molecule.state_df["j"] == j_val]
        m_len = 2 * j_val + 1
        num_states = 2 * m_len

        molecule.state_df.loc[indices_in_j[0], "state_dist"] = pop0
        molecule.state_df.loc[indices_in_j[1], "state_dist"] = pop1
        molecule.state_df.loc[indices_in_j[m_len], "state_dist"] = pop2

        # Residual population in the background is added (imperfect pumping): 5% of the total population in the j sublevel
        population_off = 0.05 * tot_pop_in_j / num_states
        molecule.state_df.loc[indices_in_j, "state_dist"] += population_off

        # Renormalization of the populations in the j manifold
        total_in_j = molecule.state_df.loc[indices_in_j, "state_dist"].sum()
        molecule.state_df.loc[indices_in_j, "state_dist"] = tot_pop_in_j * molecule.state_df.loc[indices_in_j, "state_dist"] / total_in_j

    
    modified_pop0 = np.array(modified_pop0)
    modified_pop1 = np.array(modified_pop1)
    modified_pop2 = np.array(modified_pop2)

    j_fit = np.arange(1, 51)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        j_fit = np.arange(1, 51)
        ax.plot(j_fit, modified_pop0, label="LM (target) state", color='#800020')
        ax.plot(j_fit, modified_pop1, label="PU state", color='#4B0082')
        ax.plot(j_fit, modified_pop2, label="PL state", color='#DAA520')
        ax.set_xlabel(r'Manifold $J$', fontsize=25)
        ax.set_ylabel('Population', fontsize=25)
        ax.set_title('Population after pumping and RAP stages', fontsize=28)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.legend(fontsize=21, frameon=True, loc="center left")
        ax.tick_params(axis='both', which='major', labelsize=25)
        fig.tight_layout()
        save_figure_in_images(fig, savetext)
        plt.show()

    return pop_fit_0, pop_fit_1, pop_fit_2jp1, modified_pop0, modified_pop1, modified_pop2


# ==========================================================
# CaH SPECIFIC LOGIC
# ==========================================================

def fit_populations_cah(
    popj_dict: Dict[int, float], 
    type: str, 
    no_plot: bool = True, 
    ax: Optional[plt.Axes] = None
) -> np.ndarray:
    """
    Fitting function with specific initial guesses (p0) for CaH molecules.
    """
    j_vals = np.array(sorted(popj_dict.keys()))
    pop_vals = np.array([popj_dict[j] for j in j_vals])

    def sigmoid_1_cah(J, A, B, C):
        return 0.475 / (1 + A * np.exp(-B * (J - C)))
    
    def sigmoid_2jp1_cah(J, A, B, C):
        return 0.5 - 0.5 / (1 + A * np.exp(-B * (J - C)))

    if type == "popj_1":
        p0 = [0.7, 1.3, 5.7]
        popt, _ = curve_fit(sigmoid_1_cah, j_vals, pop_vals, maxfev=5000, p0=p0)
    elif type == "popj_2jp1":
        p0 = [0.1, 1.8, 1]
        popt, _ = curve_fit(sigmoid_2jp1_cah, j_vals, pop_vals, maxfev=10000, p0=p0)
    else:
        print("Invalid type. Choose 'popj_1' or 'popj_2jp1'.")


    A_fit, B_fit, C_fit = popt
    j_fit = np.arange(1, 15, 0.25)

    if type == "popj_1":
        pop_fit = sigmoid_1_cah(j_fit, A_fit, B_fit, C_fit)
        color, label = '#4B0082', "PU state"
    elif type == "popj_2jp1":
        pop_fit = sigmoid_2jp1_cah(j_fit, A_fit, B_fit, C_fit)
        color, label = '#DAA520', "PL state"
    else:
        print("Invalid type. Choose 'popj_1' or 'popj_2jp1'.")
    

    if not no_plot:
        ax.scatter(j_vals, pop_vals, label=label, color=color)
        ax.plot(j_fit, pop_fit, label="fit " + label, color=color, linestyle='-')

    return pop_fit


def pumping_manifolds_cah(
    data_files: List[Path], 
    plot: bool = True, 
    savetext: str = "cah_after_pumping.svg"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes QuTiP density matrices to extract manifold populations for CaH.
    """
    popj_0_dict, popj_1_dict, popj_2jp1_dict = {}, {}, {}

    for file_path in data_files:
        data = joblib.load(file_path)
        j_val = data["j_val"]
        rho_final = data["rho_final"]
        args = data["args"]
        n_motional = args["terms"][0]["n_motional"]
        n_internal = args["terms"][0]["n_internal"]

        populations = np.zeros(n_internal)
        for j in range(n_internal):
            Pj_op = tensor(basis(n_internal, j) * basis(n_internal, j).dag(), qeye(n_motional))
            populations[j] = expect(Pj_op, rho_final)

        popj_0, popj_1 = populations[0], populations[1]

        # Empirical correction for CaH manifolds
        if j_val >= 8:
            popj_0 = popj_0 - 0.1 + 1 / (2 * (2 * j_val + 1)) 
            popj_1 = popj_1 + 0.1 - 1 / (2 * (2 * j_val + 1)) 

        index = min(2 * j_val + 1, int(n_internal / 2))
        popj_2jp1 = populations[index]
        popj_0_dict[j_val], popj_1_dict[j_val], popj_2jp1_dict[j_val] = popj_0, popj_1, popj_2jp1

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        pop_fit_1 = fit_populations_cah(popj_1_dict, "popj_1", no_plot=False, ax=ax)
        pop_fit_2jp1 = fit_populations_cah(popj_2jp1_dict, "popj_2jp1", no_plot=False, ax=ax)
        pop_fit_0 = 1 - pop_fit_1 - pop_fit_2jp1

        j_vals_0 = np.array(sorted(popj_0_dict.keys()))
        pop_vals_0 = np.array([popj_0_dict[j] for j in j_vals_0])
        j_fit = np.arange(1, 15, 0.25)

        ax.scatter(j_vals_0, pop_vals_0, label="LM (target) state", color='#800020')
        ax.plot(j_fit, pop_fit_0, label="fit LM state", color='#800020', linestyle='-')
        ax.set_xlabel(r'Manifold $J$', fontsize=25)
        ax.set_ylabel('Population', fontsize=25)
        ax.set_title('Population after pumping stage', fontsize=28)
        ax.grid(True)
        ax.legend(fontsize=21, loc="upper right", frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        save_figure_in_images(fig, savetext)
        plt.show()
    else:
        pop_fit_1 = fit_populations_cah(popj_1_dict, "popj_1", no_plot = True)
        pop_fit_2jp1 = fit_populations_cah(popj_2jp1_dict, "popj_2jp1", no_plot = True)
        pop_fit_0 = 1 - pop_fit_1 - pop_fit_2jp1


        j_vals_0 = np.array(sorted(popj_0_dict.keys()))
        pop_vals_0 = np.array([popj_0_dict[j] for j in j_vals_0])
        j_fit = np.arange(1, 15)

    indices_integers = np.arange(0, len(j_fit), 4)
    pop_fit_0 = pop_fit_0[indices_integers]
    pop_fit_1 = pop_fit_1[indices_integers]
    pop_fit_2jp1 = pop_fit_2jp1[indices_integers]

    return pop_fit_0, pop_fit_1, pop_fit_2jp1



def after_pumping_rap_pop_cah(
    molecule: Any, 
    state_dist: States, 
    j_max: int, 
    pop_fit_0: np.ndarray, 
    pop_fit_1: np.ndarray, 
    pop_fit_2jp1: np.ndarray,
    rap_sign_cah: np.ndarray, 
    plot: bool = True, 
    rap_flag: bool = True,
    savetext: str = "cah_after_pumping_rap_pop.svg"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies Rapid Adiabatic Passage (RAP) signatures for CaH.
    """
    thermal_distribution = state_dist.j_distribution()
    molecule.state_df.loc[molecule.state_df.index[2:], "state_dist"] = 0.0

    if not np.allclose(pop_fit_0 + pop_fit_1 + pop_fit_2jp1, 1.0, atol=1e-6):
        raise ValueError("Population is not normalized to 1")

    modified_pop0, modified_pop1, modified_pop2 = [], [], []

    for j_val in range(1, j_max + 1):

        tot_pop_in_j = thermal_distribution[j_val]

        # Pumping
        pop0 = pop_fit_0[j_val - 1] * tot_pop_in_j
        pop1 = pop_fit_1[j_val - 1] * tot_pop_in_j
        pop2 = pop_fit_2jp1[j_val - 1] * tot_pop_in_j

        if rap_flag:
            transfer = rap_sign_cah[j_val] * pop1
            pop0 = pop0 + transfer
            pop1 = pop1 - transfer

        modified_pop0.append(pop0 / tot_pop_in_j)
        modified_pop1.append(pop1 / tot_pop_in_j)
        modified_pop2.append(pop2 / tot_pop_in_j)

        states_in_j = molecule.state_df.loc[molecule.state_df["j"] == j_val].copy()
        indices_in_j = molecule.state_df.index[molecule.state_df["j"] == j_val]
        m_len = 2 * j_val + 1
        num_states = 2 * m_len

        molecule.state_df.loc[indices_in_j[0], "state_dist"] = pop0
        molecule.state_df.loc[indices_in_j[1], "state_dist"] = pop1
        molecule.state_df.loc[indices_in_j[m_len], "state_dist"] = pop2

        population_off = 0.05 * tot_pop_in_j / num_states
        molecule.state_df.loc[indices_in_j, "state_dist"] += population_off
        total_in_j = molecule.state_df.loc[indices_in_j, "state_dist"].sum()
        molecule.state_df.loc[indices_in_j, "state_dist"] = tot_pop_in_j * molecule.state_df.loc[indices_in_j, "state_dist"] / total_in_j

    modified_pop0 = np.array(modified_pop0)
    modified_pop1 = np.array(modified_pop1)
    modified_pop2 = np.array(modified_pop2)

    j_fit = np.arange(1, j_max + 1)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        j_fit = np.arange(1, j_max + 1)
        ax.plot(j_fit, modified_pop0, label="LM (target) state", color='#800020')
        ax.plot(j_fit, modified_pop1, label="PU state", color='#4B0082')
        ax.plot(j_fit, modified_pop2, label="PL state", color='#DAA520')
        ax.set_xlabel(r'Manifold $J$', fontsize=25)
        ax.set_ylabel('Population', fontsize=25)
        ax.set_title('Population after pumping and RAP stages', fontsize=28)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.legend(fontsize=21, frameon=True, loc="center left")
        ax.tick_params(axis='both', which='major', labelsize=25)
        fig.tight_layout()
        save_figure_in_images(fig, savetext)
        plt.show()

    return pop_fit_0, pop_fit_1, pop_fit_2jp1, modified_pop0, modified_pop1, modified_pop2