# File: spectrum.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Module for computing and plotting excitation spectra of molecular ions.

This module includes:
- Calculation of excitation probabilities for a molecule given laser parameters.
- Generation of the full spectrum with optional experimental imperfections.
- Functions to visualize spectra before and after pumping sequences.
- Handles noise, laser miscalibration, and false positives.
"""

import numpy as np
from molecules.molecule import CaOH, Molecule
from typing import Tuple, Optional, Dict
import exp_imperfections as imp
import matplotlib.pyplot as plt
import QLS.pumping as pumping
from saving import save_figure_in_images


def get_excitation_probabilities(
    molecule: Molecule,
    frequency: float,
    duration_us: float,
    rabi_rate_mhz: float,
    dephased: bool = False,
    coherence_time_us: float = 100.0,
    is_minus: bool = True,
    noise_params: Dict[str, Dict[str, float]] = None,
    seed: Optional[int] = None,
    laser_miscalibration: Dict[str, Dict[str, float]] = None, 
    seed_miscalibration: Optional[int] = None
) -> np.ndarray:
    """
    Returns the excitation probabilities for a molecule given a laser pulse.

    Parameters
    ----------
    molecule : Molecule
        The molecule to calculate the excitation probabilities for.
    frequency : float
        The Raman difference frequency of the excitation pulse in MHz.
    duration_us : float
        The duration of the excitation pulse in microseconds.
    rabi_rate_mhz : float
        The Rabi rate in MHz.
    dephased : bool, optional
        If True, the excitation is dephased. Default is False.
    coherence_time_us : float, optional
        The coherence time for Rabi flopping in microseconds. Default is 100.
    is_minus : bool, optional
        If True, calculates Δm = -1 transitions. Default is True.
    noise_params : dict, optional
        Dictionary specifying noise parameters for frequency and Rabi rate.
    seed : int, optional
        Seed for random noise.
    laser_miscalibration : dict, optional
        Dictionary specifying laser miscalibration parameters.
    seed_miscalibration : int, optional
        Seed for miscalibration noise.

    Returns
    -------
    np.ndarray
        The excitation probabilities for each molecular state.
    """
    if laser_miscalibration is None:
        laser_miscalibration = {}

    if "frequency" in laser_miscalibration:
        frequency = imp.apply_noise(
            frequency,
            laser_miscalibration["frequency"]["type"],
            laser_miscalibration["frequency"]["level"],
            seed_miscalibration
        )
    if "rabi_rate" in laser_miscalibration:
        rabi_rate_mhz = imp.apply_noise(
            rabi_rate_mhz,
            laser_miscalibration["rabi_rate"]["type"],
            laser_miscalibration["rabi_rate"]["level"],
            seed_miscalibration
        )

    if noise_params is None:
        noise_params = {}

    # Apply noise to frequency and Rabi rate if specified
    if "frequency" in noise_params:
        frequency = imp.apply_noise(
            frequency,
            noise_params["frequency"]["type"],
            noise_params["frequency"]["level"],
            seed
        )
    if "rabi_rate" in noise_params:
        rabi_rate_mhz = imp.apply_noise(
            rabi_rate_mhz,
            noise_params["rabi_rate"]["type"],
            noise_params["rabi_rate"]["level"],
            seed
        )

    state_exc_probs = np.zeros(len(molecule.state_df))

    # Calculate detunings for the transition
    if is_minus:
        detunings = 2 * np.pi * (frequency - molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)
    else:
        detunings = 2 * np.pi * (frequency + molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)

    omegas = rabi_rate_mhz * molecule.transition_df["coupling"].to_numpy(dtype=float)

    # Calculate excitation probabilities with or without dephasing
    if dephased:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * (
            (1 - np.cos(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us) * np.exp(-duration_us / coherence_time_us)) / 2
        )
    else:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * \
            np.sin(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us / 2) ** 2

    states_index = molecule.transition_df["index1"].to_numpy(dtype=int) if is_minus else molecule.transition_df["index2"].to_numpy(dtype=int)
    for i in range(len(molecule.transition_df)):
        state_exc_probs[states_index[i]] += transition_exc_probs[i]

    return state_exc_probs


def get_spectrum(
    molecule: Molecule,
    state_distribution: np.ndarray,
    duration_us: float,
    rabi_rate_mhz: float,
    max_frequency_mhz: float,
    scan_points: int,
    dephased: bool = True,
    coherence_time_us: float = 100.0,
    is_minus: bool = True,
    noise_params: Dict[str, Dict[str, float]] = None,
    seed: Optional[int] = None,
    laser_miscalibration: Dict[str, Dict[str, float]] = None,   
    seed_miscalibration: Optional[int] = None,
    false_positive_rate: float = 0.0,
    type_false_positive: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the excitation spectrum for a molecule over a frequency range.

    Parameters
    ----------
    molecule : Molecule
        The molecule to calculate the spectrum for.
    state_distribution : np.ndarray
        Population distribution of the states.
    duration_us : float
        Duration of the excitation pulse in microseconds.
    rabi_rate_mhz : float
        Rabi rate in MHz.
    max_frequency_mhz : float
        Maximum frequency for the scan in MHz.
    scan_points : int
        Number of frequency points.
    dephased : bool, optional
        If True, includes dephasing. Default is True.
    coherence_time_us : float, optional
        Coherence time in microseconds. Default is 100.
    is_minus : bool, optional
        If True, calculates Δm = -1 transitions. Default is True.
    noise_params : dict, optional
        Noise parameters for frequency and Rabi rate.
    seed : int, optional
        Seed for noise.
    laser_miscalibration : dict, optional
        Laser miscalibration parameters.
    seed_miscalibration : int, optional
        Seed for miscalibration noise.
    false_positive_rate : float, optional
        Probability of false excitation. Default is 0.0.
    type_false_positive : str, optional
        Type of false positive noise. Default is "uniform".

    Returns
    -------
    frequencies : np.ndarray
        Array of frequencies scanned.
    exc_probs : np.ndarray
        Excitation probabilities corresponding to each frequency.
    """
    frequencies = np.linspace(-max_frequency_mhz, max_frequency_mhz, scan_points)
    exc_probs = [
        np.dot(
            get_excitation_probabilities(
                molecule, frequency, duration_us, rabi_rate_mhz, dephased, coherence_time_us, is_minus,
                noise_params, seed, laser_miscalibration, seed_miscalibration
            ),
            state_distribution
        )
        for frequency in frequencies
    ]

    exc_probs = np.array(exc_probs)

    if false_positive_rate > 0.0:
        exc_probs = imp.false_positive_excitation(frequencies, exc_probs, false_positive_rate, type_false_positive)

    frequencies = np.array(frequencies)
    exc_probs = np.array(exc_probs)
    if np.any(exc_probs < 0):
        raise ValueError("Excitation probabilities must be non-negative")
    exc_probs = np.clip(exc_probs, 0, 1)

    return frequencies, exc_probs


def unpumped_pumped(
    b_field_gauss: float,
    j_max: int,
    cah1: Molecule,
    states,
    spectrum_list: dict,
    pump_sequences,
    scan_direction: str = "left",
    filename: Optional[str] = None,
    noise_params: Optional[dict] = None,
    laser_miscalibration: Optional[dict] = None,
    seed_miscalibration: Optional[int] = None,
    false_positive_rate: float = 0.0,
    y_lim: Optional[float] = None
):
    """
    Plot the molecular spectrum before and after pumping sequences.

    Parameters
    ----------
    b_field_gauss : float
        Magnetic field in Gauss.
    j_max : int
        Maximum rotational level.
    cah1 : Molecule
        Molecule object (CaH or CaOH).
    states : object
        Object containing state distribution (`states.dist`) and `j_distribution()` method.
    spectrum_list : dict
        Dictionary with spectrum parameters (duration_us, rabi_rate_mhz, max_frequency_mhz, scan_points, etc.).
    pump_sequences : list or dict
        Pumping sequences specifying frequency, number of pulses, duration, rabi_rate, dephased, is_minus.
    scan_direction : str, optional
        "left", "right", or "both". Default is "left".
    filename : str, optional
        Filename to save the figure. Default is None.
    noise_params : dict, optional
        Noise parameters for pumping.
    laser_miscalibration : dict, optional
        Laser miscalibration parameters.
    seed_miscalibration : int, optional
        Seed for miscalibration noise.
    false_positive_rate : float, optional
        Probability of false excitation. Default is 0.0.
    y_lim : float, optional
        Maximum value of y-axis. Default is None.
    """

    if isinstance(pump_sequences, dict):
        pump_sequences = [pump_sequences]

    signature_transitions = np.array([cah1.transition_df.loc[cah1.transition_df["j"]==j].iloc[0]["energy_diff"] * 1e-3 for j in range(1, cah1.j_max+1)])


    # Spectrum before pumping.
    frequencies, exc_probs1_before = get_spectrum(cah1, states.dist, **spectrum_list, 
                                                  noise_params=noise_params, seed=None, laser_miscalibration=laser_miscalibration,
                                                  seed_miscalibration=seed_miscalibration,
                                                  false_positive_rate=false_positive_rate, type_false_positive="uniform")

    # Pumping
    for seq in pump_sequences:

        pump_frequency_mhz = seq["frequency_mhz"]
        num_pumps = seq["num_pumps"]
        pump_duration_us = seq["duration_us"]
        pump_rabi_rate_mhz = seq["rabi_rate_mhz"]
        pump_dephased = seq["dephased"]
        is_minus = seq["is_minus"]

        for _ in range(num_pumps):
            states.dist += pumping.excitation_matrix(cah1, pump_frequency_mhz, pump_duration_us, pump_rabi_rate_mhz, pump_dephased, is_minus,
                                                    noise_params=noise_params, seed=None, laser_miscalibration=laser_miscalibration,
                                                    seed_miscalibration=seed_miscalibration).dot(states.dist)


    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(frequencies, exc_probs1_before, label = "Unpumped", linestyle = "-", color = "blue", linewidth=3)     

    # Pumped spectrum
    frequencies, exc_probs1_after = get_spectrum(cah1, states.dist, **spectrum_list, 
                                                 noise_params=noise_params, seed=None, laser_miscalibration=laser_miscalibration,
                                                 seed_miscalibration=seed_miscalibration,
                                                 false_positive_rate=false_positive_rate, type_false_positive="uniform")

    ax.plot(frequencies, exc_probs1_after, label = f"Pumped", linestyle = "-", color = "orange", linewidth=3)      



    ax.bar(signature_transitions, states.j_distribution()[1:], color="red", width=0.0005, alpha=0.1, label = "Sign. Trans.")

    for seq in pump_sequences:
        pump_frequency_mhz = seq["frequency_mhz"]
        ax.axvline(x=pump_frequency_mhz, color='black', linestyle='--', label=f'Pump freq {pump_frequency_mhz*(1e3)} kHz', linewidth=3)


    ax.legend(frameon=False, fontsize=25)
    ax.set_xlabel("Frequency (MHz)", fontsize=25)
    ax.set_ylabel("Excitation probability", fontsize=25)
    title = "CaH$^+$ hyperfine spectrum at B=" + str(b_field_gauss) + " G"

    ax.set_title(title, fontsize=28)

    max_frequency_mhz = spectrum_list["max_frequency_mhz"]


    if scan_direction == "right":
        a_lim = 0
        b_lim = max_frequency_mhz
    elif scan_direction == "left":
        a_lim = -max_frequency_mhz
        b_lim = 0
    elif scan_direction == "both":
        a_lim = -max_frequency_mhz
        b_lim = max_frequency_mhz
    else:
        raise ValueError("scan_direction must be 'right', 'left', or 'both'")

    ax.set_xlim([a_lim, b_lim])

    if y_lim is None:
        if isinstance(cah1, CaOH):
            ax.set_ylim([0, 0.18])
        else:
            ax.set_ylim([0, 0.225])
    else:
        ax.set_ylim([0, y_lim])

    ax.tick_params(axis='both', which='major', labelsize=25)

    if not isinstance(cah1, CaOH):
        for i in range(j_max):
            if signature_transitions[i] >= a_lim and signature_transitions[i] <= b_lim:
                ax.text(signature_transitions[i], states.j_distribution()[i+1], f"J={i+1}", ha='center', va='bottom', color='red', alpha=0.8, fontsize=25)

    if filename is not None:
        save_figure_in_images(fig, filename)
    else:
        save_figure_in_images(fig, "unpumpedcah.svg")

    plt.show()


def spectrum_w_wo_imperfections(
    b_field_gauss: float,
    j_max: int,
    cah1: Molecule,
    states,
    spectrum_list: dict,
    scan_direction: str = "left",
    pump_sequences=None,
    filename: Optional[str] = None,
    noise_params: Optional[dict] = None,
    laser_miscalibration: Optional[dict] = None,
    seed_miscalibration: Optional[int] = None,
    false_positive_rate: float = 0.0,
    y_lim: Optional[float] = None
):
    """
    Plot the molecular spectrum with and without experimental imperfections.

    Parameters
    ----------
    b_field_gauss : float
        Magnetic field in Gauss.
    j_max : int
        Maximum rotational level.
    cah1 : Molecule
        Molecule object (CaH or CaOH).
    states : object
        Object containing state distribution (`states.dist`) and `j_distribution()` method.
    spectrum_list : dict
        Dictionary with spectrum parameters (duration_us, rabi_rate_mhz, max_frequency_mhz, scan_points, etc.).
    scan_direction : str, optional
        "left", "right", or "both". Default is "left".
    pump_sequences : list or dict, optional
        Pumping sequences specifying frequency, number of pulses, duration, rabi_rate, dephased, is_minus.
    filename : str, optional
        Filename to save the figure. Default is None.
    noise_params : dict, optional
        Noise parameters for pumping or laser miscalibration.
    laser_miscalibration : dict, optional
        Laser miscalibration parameters.
    seed_miscalibration : int, optional
        Seed for miscalibration noise.
    false_positive_rate : float, optional
        Probability of false excitation. Default is 0.0.
    y_lim : float, optional
        Maximum value of y-axis. Default is None.
    """

    if pump_sequences is not None and isinstance(pump_sequences, dict):
        pump_sequences = [pump_sequences]

    signature_transitions = np.array([cah1.transition_df.loc[cah1.transition_df["j"]==j].iloc[0]["energy_diff"] * 1e-3 for j in range(1, cah1.j_max+1)])

    # Pumping if specified
    if pump_sequences is not None:
        for seq in pump_sequences:

            pump_frequency_mhz = seq["frequency_mhz"]
            num_pumps = seq["num_pumps"]
            pump_duration_us = seq["duration_us"]
            pump_rabi_rate_mhz = seq["rabi_rate_mhz"]
            pump_dephased = seq["dephased"]
            is_minus = seq["is_minus"]

            for _ in range(num_pumps):
                states.dist += pumping.excitation_matrix(cah1, pump_frequency_mhz, pump_duration_us, pump_rabi_rate_mhz, pump_dephased, is_minus,
                                                        noise_params=noise_params, seed=None, laser_miscalibration=laser_miscalibration,
                                                        seed_miscalibration=seed_miscalibration).dot(states.dist)


    # Spectrum with and without imperfections
    if laser_miscalibration is not None or noise_params is not None or false_positive_rate > 0.0:
        frequencies, exc_probs_imperfections = get_spectrum(molecule=cah1, state_distribution=states.dist, **spectrum_list, 
                                                    noise_params=noise_params, seed=None, laser_miscalibration=laser_miscalibration,
                                                    seed_miscalibration=seed_miscalibration,
                                                    false_positive_rate=false_positive_rate, type_false_positive="uniform")

    frequencies, exc_probs_no_imperfections = get_spectrum(molecule=cah1, state_distribution=states.dist, **spectrum_list, 
                                                    noise_params=None, seed=None, laser_miscalibration=None,
                                                    seed_miscalibration=None,
                                                    false_positive_rate=0.0, type_false_positive="uniform")


    fig, ax = plt.subplots(figsize=(14, 8))

    if laser_miscalibration is not None or noise_params is not None or false_positive_rate > 0.0:
        ax.plot(frequencies, exc_probs_imperfections, label = "w/ imperfections", linestyle = "-", color = "blue", linewidth=3) 
       
    ax.plot(frequencies, exc_probs_no_imperfections, label = "w/o imperfections", linestyle = "-", color = "orange", linewidth=3)     
    ax.bar(signature_transitions, states.j_distribution()[1:], color="red", width=0.0005, alpha=0.1, label = "Sign. Trans.")

    if pump_sequences is not None:
        for seq in pump_sequences:
            pump_frequency_mhz = seq["frequency_mhz"]
            ax.axvline(x=pump_frequency_mhz, color='black', linestyle='--', label=f'Pump freq {pump_frequency_mhz*(1e3)} kHz', linewidth=3)

    ax.legend(frameon=False, fontsize=25)
    ax.set_xlabel("Frequency (MHz)", fontsize=25)
    ax.set_ylabel("Excitation probability", fontsize=25)
    title = "CaH$^+$ hyperfine spectrum at B=" + str(b_field_gauss) + " G"

    ax.set_title(title, fontsize=28)

    max_frequency_mhz = spectrum_list["max_frequency_mhz"]


    if scan_direction == "right":
        a_lim = 0
        b_lim = max_frequency_mhz
    elif scan_direction == "left":
        a_lim = -max_frequency_mhz
        b_lim = 0
    elif scan_direction == "both":
        a_lim = -max_frequency_mhz
        b_lim = max_frequency_mhz
    else:
        raise ValueError("scan_direction must be 'right', 'left', or 'both'")

    ax.set_xlim([a_lim, b_lim])

    if y_lim is None:
        if isinstance(cah1, CaOH):
            ax.set_ylim([0, 0.18])
        else:
            ax.set_ylim([0, 0.225])
    else:
        ax.set_ylim([0, y_lim])
    
    ax.tick_params(axis='both', which='major', labelsize=25)

    if not isinstance(cah1, CaOH):
        for i in range(j_max):
            if signature_transitions[i] >= a_lim and signature_transitions[i] <= b_lim:
                ax.text(signature_transitions[i], states.j_distribution()[i+1], f"J={i+1}", ha='center', va='bottom', color='red', alpha=0.8, fontsize=25)

    if filename is not None:
        save_figure_in_images(fig, filename)
    else:
        save_figure_in_images(fig, "unpumpedcah.svg")

    plt.show()

