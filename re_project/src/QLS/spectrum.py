import numpy as np
from molecules.molecule import CaOH, Molecule, CaH
from scipy.constants import h, k
from scipy.sparse import csr_array, sparray
from typing import Tuple, Optional, NamedTuple
from typing import List, Dict
import exp_imperfections as imp
import matplotlib.pyplot as plt
import QLS.state_dist as state_dist
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
    seed: int = None,
    laser_miscalibration: Dict[str, Dict[str, float]] = None, 
    seed_miscalibration: int = None
) -> np.ndarray:
    """Returns the excitation probabilities for given frequency and other parameters

    Args:
        molecule (Molecule): The molecule to calculate the excitation probabilities for
        frequency (float): The frequency of the excitation pulse in MHz
        duration_us (float): The duration of the excitation pulse in microseconds
        rabi_rate_mhz (float): The Rabi rate in MHz
        dephased (bool): If True, the excitation is dephased
        coherence_time_us (float): The coherence time in us for rabi flopping
        is_minus (bool): If True, the excitation is for dm = -1
    Returns:
        np.ndarray: The excitation probabilities for each state
    """


    if laser_miscalibration is None:
        laser_miscalibration = {}

    if "frequency" in laser_miscalibration:
        frequency = imp.apply_noise(frequency, laser_miscalibration["frequency"]["type"], laser_miscalibration["frequency"]["level"], seed_miscalibration)
    if "rabi_rate" in laser_miscalibration:
        rabi_rate_mhz = imp.apply_noise(rabi_rate_mhz, laser_miscalibration["rabi_rate"]["type"], laser_miscalibration["rabi_rate"]["level"], seed_miscalibration)


    if noise_params is None:
        noise_params = {}

    # Applicare rumore separato per ogni parametro se specificato
    if "frequency" in noise_params:
        frequency = imp.apply_noise(frequency, noise_params["frequency"]["type"], noise_params["frequency"]["level"], seed)
    if "rabi_rate" in noise_params:
        rabi_rate_mhz = imp.apply_noise(rabi_rate_mhz, noise_params["rabi_rate"]["type"], noise_params["rabi_rate"]["level"], seed)


    state_exc_probs = np.zeros(len(molecule.state_df))

    if is_minus:
        detunings = 2 * np.pi * (frequency - molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)
    else:
        detunings = 2 * np.pi * (frequency + molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)


    # detunings = 2 * np.pi * (frequency - molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)
    omegas = rabi_rate_mhz * molecule.transition_df["coupling"].to_numpy(dtype=float)

    if dephased:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * ((1 - np.cos(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us) * np.exp(-duration_us / coherence_time_us)) / 2)
    else:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * np.sin(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us / 2) ** 2

    if is_minus:
        # state1 --> state2
        states_index = molecule.transition_df["index1"].to_numpy(dtype=int)
    else:
        # state2 --> state1
        states_index = molecule.transition_df["index2"].to_numpy(dtype=int)

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
    seed: int = None,
    laser_miscalibration: Dict[str, Dict[str, float]] = None,   
    seed_miscalibration: int = None,
    false_positive_rate: float = 0.0,
    type_false_positive: str = "uniform"

) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the spectrum for given parameters

    Args:
        molecule (Molecule): The molecule to calculate the spectrum for
        duration_us (float): The duration of the excitation pulse in microseconds
        rabi_rate_mhz (float): The Rabi rate in MHz
        max_frequency_mhz (float): The maximum frequency in MHz
        scan_points (int): The number of scan points
        dephased (bool): If True, the excitation is dephased
        is_minus (bool): If True, the excitation is for dm = -1; otherwise, dm = +1
    Returns:
        np.ndarray: The excitation probabilities for each frequency
    """
    frequencies = np.linspace(-max_frequency_mhz, max_frequency_mhz, scan_points)
    exc_probs = [
        np.dot(
            get_excitation_probabilities(molecule, frequency, duration_us, rabi_rate_mhz, dephased, coherence_time_us, is_minus, noise_params, seed, laser_miscalibration, seed_miscalibration),
            state_distribution,
        )
        for frequency in frequencies
    ]

    exc_probs = np.array(exc_probs)

    if false_positive_rate > 0.0:
        exc_probs = imp.false_positive_excitation(frequencies, exc_probs, false_positive_rate, type_false_positive)

    # Convert to numpy array
    frequencies = np.array(frequencies)
    exc_probs = np.array(exc_probs)

    # Ensure the excitation probabilities are non-negative
    if np.any(exc_probs < 0):
        raise ValueError("Excitation probabilities must be non-negative")
    
    # Ensure the excitation probabilities are in the range [0, 1]
    exc_probs = np.clip(exc_probs, 0, 1)

    return frequencies, exc_probs





def unpumped_pumped(b_field_gauss, j_max, cah1, states, spectrum_list, pump_sequences, scan_direction = "left", filename = None, noise_params=None, laser_miscalibration=None, seed_miscalibration=None, false_positive_rate=0.0, y_lim=None):

    # I compute the states1. I do so because i need states1.dist,          J|m|csi|...|states1.dist
                                                                        #  .|.| . |   |     .
                                                                        #  .|.| . |   |     .
                                                                        #  .|.| . |   |     .

    if isinstance(pump_sequences, dict):
        pump_sequences = [pump_sequences]



    # I take the transition_df. For each j (multiplet), i take the energy difference of the target distribution.
    signature_transitions = np.array([cah1.transition_df.loc[cah1.transition_df["j"]==j].iloc[0]["energy_diff"] * 1e-3 for j in range(1, cah1.j_max+1)])



    # get_spectrum: it takes excitation probability (from get_excitation_probabilities) + states1.dist distribution (from States) and np.dot @ freq.
    # Then repeats for the frequencies and returns both the frequencies and the results @ each freq.
    # This returns the spectrum before pumping.
    frequencies, exc_probs1_before = get_spectrum(cah1, states.dist, **spectrum_list, 
                                                           noise_params=noise_params, seed=None, laser_miscalibration=laser_miscalibration,
                                                           seed_miscalibration=seed_miscalibration,
                                                           false_positive_rate=false_positive_rate, type_false_positive="uniform")


    # I update the state distribution (@ fixed j) with the exctiation matrix.
    # I pump the system multiple times in order to better populate the molecule

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
    ax.plot(frequencies, exc_probs1_before, label = "Unpumped", linestyle = "-", color = "blue", linewidth=3)     # Plot before pumping

    frequencies, exc_probs1_after = get_spectrum(cah1, states.dist, **spectrum_list, 
                                                           noise_params=noise_params, seed=None, laser_miscalibration=laser_miscalibration,
                                                           seed_miscalibration=seed_miscalibration,
                                                           false_positive_rate=false_positive_rate, type_false_positive="uniform")

    ax.plot(frequencies, exc_probs1_after, label = f"Pumped", linestyle = "-", color = "orange", linewidth=3)      # Plot after pumping



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





def spectrum_w_wo_imperfections(b_field_gauss, j_max, cah1, states, spectrum_list, scan_direction = "left", pump_sequences = None, filename = None, noise_params=None, laser_miscalibration=None, seed_miscalibration=None, false_positive_rate=0.0, y_lim=None):

    # I compute the states1. I do so because i need states1.dist,          J|m|csi|...|states1.dist
                                                                        #  .|.| . |   |     .
                                                                        #  .|.| . |   |     .
                                                                        #  .|.| . |   |     .

    if pump_sequences is not None and isinstance(pump_sequences, dict):
        pump_sequences = [pump_sequences]


    # I take the transition_df. For each j (multiplet), i take the energy difference of the target distribution.
    signature_transitions = np.array([cah1.transition_df.loc[cah1.transition_df["j"]==j].iloc[0]["energy_diff"] * 1e-3 for j in range(1, cah1.j_max+1)])


    # I update the state distribution (@ fixed j) with the exctiation matrix.
    # I pump the system multiple times in order to better populate the molecule
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


    # get_spectrum: it takes excitation probability (from get_excitation_probabilities) + states1.dist distribution (from States) and np.dot @ freq.
    # Then repeats for the frequencies and returns both the frequencies and the results @ each freq.
    # This returns the spectrum before pumping.

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
