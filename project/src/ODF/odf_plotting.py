"""
Visualization Module for Molecular and Atomic Evolutionary Dynamics.

This module provides a suite of plotting functions to visualize:
- Phonon number evolution and spin populations in 3-level molecular systems.
- Experimental vs. simulated Optical Dipole Force (ODF) data.
- Phase space trajectories (X and P quadratures).
- Spectral comparisons under varying magnetic fields.
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Optional, List, Tuple, Any

from saving import save_figure_in_images
from utils import find_main_peaks
from QLS.spectrum import get_spectrum
from QLS.state_dist import *
from molecules.molecule import CaOH
from ODF.data_fitting import sinc


# =================================================================================
# MOLECULE + MOTION + ION (3-LEVEL SYSTEM)
# =================================================================================

def plot_atom_mol3_evolution(
    x_data: np.ndarray,
    y_data: np.ndarray,
    fin_time: float,
    w_mol: float,
    rabi_rate_molecule: float,
    rabi_rate: float,
    detunings: np.ndarray,
    exc_mol_odf: List[float],
    exc_mol_bsb: List[float],
    exc_mol_tot: List[float],
    spin_up_atom_odf: List[float],
    spin_up_atom_bsb: List[float],
    spin_up_atom_tot: List[float],
    B: Optional[float] = None
) -> None:
    """
    Plots the comparative evolution of a 3-level molecule and its atomic readout.

    Parameters
    ----------
    x_data : np.ndarray
        Experimental frequency data points.
    y_data : np.ndarray
        Experimental excitation probability data.
    fin_time : float
        Final simulation time in microseconds.
    w_mol : float
        Molecular transition frequency in MHz.
    rabi_rate_molecule : float
        Rabi rate of the molecular transition.
    rabi_rate : float
        Rabi rate associated with the ODF.
    detunings : np.ndarray
        Array of detunings used in the simulation (rad/s).
    exc_mol_odf : list
        Phonon excitation values for the ODF Hamiltonian.
    exc_mol_bsb : list
        Phonon excitation values for the Blue Sideband Hamiltonian.
    exc_mol_tot : list
        Phonon excitation values for the combined (Total) Hamiltonian.
    spin_up_atom_odf : list
        Atomic spin-up population for ODF.
    spin_up_atom_bsb : list
        Atomic spin-up population for BSB.
    spin_up_atom_tot : list
        Atomic spin-up population for Total Hamiltonian.
    B : float, optional
        Magnetic field strength in Gauss. Default is None.
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Detuning conversion factor from rad/s to kHz for the x-axis
    detuning_khz = detunings / (2 * np.pi * 1e-3)

    # --- Subplot 1: Average Phonon Number (<n>) ---
    ax[0].plot(detuning_khz, exc_mol_bsb, color='black', label='BSB', linestyle='-', linewidth=3)
    ax[0].plot(detuning_khz, exc_mol_odf, color='black', label='ODF', linestyle=':', linewidth=3)
    ax[0].plot(detuning_khz, exc_mol_tot, color='red', label='BSB+ODF', linestyle='--', linewidth=3)
    ax[0].set_xlabel("Detuning (kHz)", fontsize=25)
    ax[0].set_ylabel(r"$\langle n \rangle$", fontsize=25)
    ax[0].set_title("Average phonon number", fontsize=28)
    ax[0].legend(frameon=True)
    ax[0].grid(True, which='major', linestyle='--', alpha=0.5)
    ax[0].tick_params(axis='both', which='major', labelsize=25)
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=5))

    # --- Subplot 2: Comparison with ODF Experimental Data ---
    ax[1].plot(detuning_khz, spin_up_atom_odf, color='black', label='ODF', linestyle=':', linewidth=3)
    ax[1].scatter(x_data / (1e-3), y_data, color='red', label='Data') 
    ax[1].set_xlabel("Detuning (kHz)", fontsize=25)
    ax[1].set_ylabel("Excitation probability", fontsize=25)
    ax[1].set_title("ODF experimental data", fontsize=28)
    ax[1].legend(frameon=True)
    ax[1].grid(True, which='major', linestyle='--', alpha=0.5)
    ax[1].tick_params(axis='both', which='major', labelsize=25)
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=5))

    # --- Subplot 3: Red Sideband (RSB) Readout Evolution ---
    ax[2].plot(detuning_khz, spin_up_atom_bsb, color='black', label='BSB', linestyle='-', linewidth=3)
    ax[2].plot(detuning_khz, spin_up_atom_odf, color='black', label='ODF', linestyle=':', linewidth=3)
    ax[2].plot(detuning_khz, spin_up_atom_tot, color='red', label='BSB+ODF', linestyle='--', linewidth=3)
    ax[2].set_xlabel("Detuning (kHz)", fontsize=25)
    ax[2].set_ylabel("Excitation probability", fontsize=25)
    ax[2].set_title("RSB readout", fontsize=28)
    ax[2].legend(frameon=True)
    ax[2].grid(True, which='major', linestyle='--', alpha=0.5)
    ax[2].tick_params(axis='both', which='major', labelsize=25)
    ax[2].xaxis.set_major_locator(MaxNLocator(nbins=5))

    # Construct the title and filename based on the presence of the B-field
    if B is None:
        fig.suptitle(
            f"mol trans = {w_mol*1e3:.2f} kHz. Duration={fin_time/1e3:.2f} ms, "
            f"RR_mol={rabi_rate_molecule:.6f}; RR_odf={rabi_rate:.6f}",
            fontsize=12
        )
        title = f"odf_rrmol{rabi_rate_molecule:.6f}_rr{rabi_rate:.6f}.svg"
    else:
        fig.suptitle(
            f"B={B} G, mol trans = {w_mol*1e3:.2f} kHz. Duration={fin_time/1e3:.2f} ms, "
            f"RR_mol={rabi_rate_molecule:.6f}; RR_odf={rabi_rate:.6f}",
            fontsize=12
        )
        title = f"odf_b{B}_rrmol{rabi_rate_molecule:.6f}_rr{rabi_rate:.6f}.svg"

    fig.tight_layout()
    save_figure_in_images(fig, title)
    plt.show()


def plot_odf_3mol_data(
    detunings: np.ndarray,
    spin_up_atom_odf: List[float],
    x_data: np.ndarray,
    y_data: np.ndarray
) -> None:
    """
    Specific plot comparing simulated ODF signals with 3-level molecular data.

    Parameters
    ----------
    detunings : np.ndarray
        Simulation detuning array.
    spin_up_atom_odf : list
        Calculated atomic spin-up probability.
    x_data : np.ndarray
        Experimental frequencies (Hz).
    y_data : np.ndarray
        Experimental excitation values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot simulation results as a dotted line
    ax.plot(
        detunings / (2 * np.pi * 1e-3), 
        spin_up_atom_odf, 
        color='black', 
        label='Simulated ODF', 
        linestyle=':', 
        linewidth=3
    )
    
    # Plot experimental data markers
    ax.plot(
        x_data / (1e-3),
        y_data,
        color='red',
        marker='^',
        markersize=10,
        linewidth=2,
        label='Data'
    )

    ax.set_xlabel("Detuning (kHz)", fontsize=25)
    ax.set_ylabel("Excitation probability", fontsize=25)
    ax.set_title("ODF experimental data", fontsize=28)
    ax.legend(frameon=True, fontsize=25)
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout()
    save_figure_in_images(fig, "ODF_experimental_data.pdf")
    plt.show()


# =================================================================================
# MOLECULE + MOTION + ION (STANDARD SYSTEM)
# =================================================================================

def plot_atom_mol_evolution(
    w_mol: float,
    rabi_rate_molecule: float,
    rabi_rate: float,
    detunings: np.ndarray,
    exc_mol_odf: List[float],
    exc_mol_bsb: List[float],
    exc_mol_tot: List[float],
    spin_up_mol_odf: List[float],
    spin_up_mol_bsb: List[float],
    spin_up_mol_tot: List[float],
    spin_up_atom_odf: List[float],
    spin_up_atom_bsb: List[float],
    spin_up_atom_tot: List[float]
) -> None:
    """
    Plots the evolutionary dynamics of the molecule-atom system in the frequency domain.

    Parameters
    ----------
    w_mol : float
        Molecular transition frequency (MHz).
    rabi_rate_molecule : float
        Rabi rate for the molecule.
    rabi_rate : float
        Rabi rate for the ODF.
    detunings : np.ndarray
        Simulation detunings.
    exc_mol_... : list
        Phonon number arrays for ODF, BSB, and Total Hamiltonians.
    spin_up_mol_... : list
        Molecular spin-up populations.
    spin_up_atom_... : list
        Atomic spin-up populations after readout.
    """
    plt.figure(figsize=(18, 4))
    detuning_mhz = detunings / (2 * np.pi)

    # --- Subplot 1: Molecular Phonon Excitation ---
    plt.subplot(1, 3, 1)
    plt.plot(detuning_mhz, exc_mol_bsb, color='black', label='BSB')
    plt.plot(detuning_mhz, exc_mol_odf, color='black', label='ODF', linestyle=':')
    plt.plot(detuning_mhz, exc_mol_tot, color='red', label='BSB+ODF', linestyle='--')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("<n>")
    plt.title(f"MOLECULE <n>")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()

    # --- Subplot 2: Molecular Spin Population ---
    plt.subplot(1, 3, 2)
    plt.plot(detuning_mhz, spin_up_mol_bsb, color='black', label='BSB')
    plt.plot(detuning_mhz, spin_up_mol_odf, color='black', label='ODF', linestyle=':')
    plt.plot(detuning_mhz, spin_up_mol_tot, color='red', label='BSB+ODF', linestyle='--')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("|<spin_up>|^2")
    plt.title(f"MOLECULE spin_up")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()

    # --- Subplot 3: Atomic Readout Population ---
    plt.subplot(1, 3, 3)
    plt.plot(detuning_mhz, spin_up_atom_bsb, color='black', label='BSB')
    plt.plot(detuning_mhz, spin_up_atom_odf, color='black', label='ODF', linestyle=':')
    plt.plot(detuning_mhz, spin_up_atom_tot, color='red', label='BSB+ODF', linestyle='--')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("|<spin_up>|^2")
    plt.title(f"RSB readout with pi-time")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()

    plt.suptitle(
        f"|spin_down>|0>|spin_down>. mol trans = {w_mol*1e3}kHz. "
        f"RR_mol={rabi_rate_molecule:.6f}; RR_odf={rabi_rate:.6f}", 
        fontsize=12
    )
    plt.tight_layout()
    plt.show()


# =================================================================================
# MOLECULE + MOTION (PHASE SPACE AND COMPARISON)
# =================================================================================

def odf_peak_phase_space(
    x_vals: List[float],
    p_vals: List[float],
    detunings: np.ndarray,
    exc: List[float]
) -> None:
    """
    Visualizes the ODF peak and the corresponding final states in phase space.

    Parameters
    ----------
    x_vals : list
        Expectation values of the X quadrature.
    p_vals : list
        Expectation values of the P quadrature.
    detunings : np.ndarray
        Simulation detunings.
    exc : list
        Phonon excitation values.
    """
    plt.figure(figsize=(12, 5))
    detuning_mhz = detunings / (2 * np.pi)

    # Spectrum plot
    plt.subplot(1, 2, 1)
    plt.plot(detuning_mhz, exc, marker='o', linestyle='-', color='blue')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("<n>")
    plt.title("ODF Peak Scan")

    # Quadrature plot (Phase Space)
    plt.subplot(1, 2, 2)
    plt.scatter(x_vals, p_vals, c=detuning_mhz, cmap='viridis', edgecolors='k')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.colorbar(label="Detuning (MHz)")
    plt.xlabel("X quadrature (<x>)")
    plt.ylabel("P quadrature (<p>)")
    plt.title("Final Phase space state")

    plt.tight_layout()
    plt.show()


def plot_mol_evolution_phase_space(
    detunings: np.ndarray,
    exc: List[float],
    prob: List[float],
    spin_up: List[float],
    x_vals: List[float],
    p_vals: List[float],
    w_mol: float
) -> None:
    """
    Provides a three-panel view of excitation probabilities and phase space.

    Parameters
    ----------
    detunings : np.ndarray
        Array of detunings.
    exc : list
        Phonon excitation values.
    prob : list
        Excitation probability (1 - ground state population).
    spin_up : list
        Molecular spin-up population.
    x_vals, p_vals : list
        Quadrature components.
    w_mol : float
        Molecular transition frequency.
    """
    plt.figure(figsize=(18, 5))
    detuning_mhz = detunings / (2 * np.pi)

    # Subplot 1: Expectation values vs Detuning
    plt.subplot(1, 3, 1)
    plt.plot(detuning_mhz, exc, marker='o', linestyle='-', label="BSB+ODF - <n>", color='blue')
    plt.plot(detuning_mhz, spin_up, label="BSB+ODF - |<spin_up>|^2", color='red', linestyle="-")
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("average values")
    plt.title(f"|spin_down>|0>, mol trans = {w_mol*1e3}kHz")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1, label=f"{w_mol*1e3}kHz")
    plt.legend(loc="upper right", fontsize=12)

    # Subplot 2: Transition Probability
    plt.subplot(1, 3, 2)
    plt.plot(detuning_mhz, prob, marker='o', linestyle='-', label="BSB+ODF - 1-|<0|0>|^2", color="blue")
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("excitation probability")
    plt.title(f"|spin_down>|0>, mol trans = {w_mol*1e3}kHz")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1, label=f"{w_mol*1e3}kHz")
    plt.legend(loc="upper right", fontsize=12)

    # Subplot 3: Phase Space scatter plot
    plt.subplot(1, 3, 3)
    plt.scatter(x_vals, p_vals, c=detuning_mhz, cmap='viridis', edgecolors='k')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.colorbar(label="Detuning (MHz)")
    plt.xlabel("X quadrature (<x>)")
    plt.ylabel("P quadrature (<p>)")
    plt.title("Final Phase space state")

    plt.tight_layout()
    plt.show()


def odf_peak_comparison(
    exc_bsb: List[float],
    exc_tot: List[float],
    spin_up_bsb: List[float],
    spin_up_tot: List[float],
    detunings: np.ndarray,
    w_mol: float
) -> None:
    """
    Compares the pure BSB excitation against the Total (BSB + ODF) Hamiltonian.

    Parameters
    ----------
    exc_bsb, exc_tot : list
        Phonon number arrays.
    spin_up_bsb, spin_up_tot : list
        Molecular spin-up population arrays.
    detunings : np.ndarray
        Array of detunings.
    w_mol : float
        Molecular transition frequency.
    """
    plt.figure(figsize=(12, 5))
    detuning_mhz = detunings / (2 * np.pi)

    # Panel 1: Phonon Number comparison
    plt.subplot(1, 2, 1)
    plt.plot(detuning_mhz, exc_bsb, color='black', label='BSB', linestyle='-')
    plt.plot(detuning_mhz, exc_tot, color='red', label='BSB+ODF', linestyle='--')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("<n>")
    plt.title(f"|spin_down>|0>, mol trans = {w_mol*1e3}kHz \n <n>")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()

    # Panel 2: Spin population comparison
    plt.subplot(1, 2, 2)
    plt.plot(detuning_mhz, spin_up_bsb, color='black', label='BSB', linestyle='-')
    plt.plot(detuning_mhz, spin_up_tot, color='red', label='BSB+ODF', linestyle='--')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("|<spin_up>|^2")
    plt.title(f"|spin_down>|0>, mol trans = {w_mol*1e3}kHz \n spin_up")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # Resetting plotting style for following functions
    plt.style.use(['science', 'notebook', 'ieee'])
    plt.rcParams['font.family'] = 'Times New Roman'


# =================================================================================
# CLASSICAL ODF SPECTROSCOPY
# =================================================================================

def classical_odf_varying_B(
    b_field_gauss: List[float],
    j_max: int,
    temperature: float,
    spectrum_list: dict,
    result: Any
) -> Tuple[List[float], List[float]]:
    """
    Simulates and plots molecular spectra for a range of magnetic field values.

    Parameters
    ----------
    b_field_gauss : list
        List of magnetic field strengths to simulate.
    j_max : int
        Maximum rotational quantum number.
    temperature : float
        Molecular sample temperature.
    spectrum_list : dict
        Parameters for the get_spectrum function.
    result : Any
        Fitting result object containing sinc parameters.

    Returns
    -------
    tuple
        (pos_peaks, neg_peaks) containing the main peak frequencies found.
    """
    plt.style.use('default')

    pos_peaks = []
    neg_peaks = []

    # Setup color mapping based on magnetic field strength
    cmap = plt.get_cmap("plasma") 
    norm = plt.Normalize(vmin=min(b_field_gauss), vmax=max(b_field_gauss))
    colors = [cmap(norm(b)) for b in b_field_gauss]

    fig, axs = plt.subplots(4, 1, figsize=(12, 18))

    for b, color in zip(b_field_gauss, colors):
        # Generate molecule and state distribution for each B field
        mo1 = CaOH.create_molecule_data(b_field_gauss=b, j_max=j_max)
        states1 = States(mo1, temperature)
        
        frequencies, exc_probs = get_spectrum(
            molecule=mo1,
            state_distribution=mo1.state_df["state_dist"],
            **spectrum_list,
            noise_params=None,
            seed=None
        )

        # Apply the ODF signal (false positive) to the raw spectrum
        odf_false_positive = sinc(frequencies, **result.best_values)
        exc_probs_odf = exc_probs + odf_false_positive
        
        # Locate peaks in the spectrum
        freq_neg_peak, freq_pos_peak = find_main_peaks(frequencies, exc_probs)
        neg_peaks.append(freq_neg_peak)
        pos_peaks.append(freq_pos_peak)

        # Plotting various views: with/without ODF signals
        axs[0].plot(frequencies, exc_probs_odf, color=color, linestyle='-', label=f"w/ODF; B={b} G")
        axs[0].plot(frequencies, exc_probs, color=color, linestyle='--') 

        axs[1].plot(frequencies, exc_probs_odf, color=color, linestyle='-', label=f"w/ODF; B={b} G")
        axs[2].plot(frequencies, exc_probs, color=color, linestyle='--', label=f"w/ODF; B={b} G")

    # Layout styling for subplots 0-2
    for ax in axs[0:3]:
        ax.set_xlim([-spectrum_list['max_frequency_mhz'], spectrum_list['max_frequency_mhz']])
        ax.set_xlim(-0.004, 0.004)
        ax.set_ylabel("Excitation probability")

    # Add the common ODF peak profile as reference
    axs[0].plot(frequencies, odf_false_positive, color='black', linestyle=':', label='ODF peak')  
    axs[1].plot(frequencies, odf_false_positive, color='black', linestyle=':', label='ODF peak')  
    axs[2].plot(frequencies, odf_false_positive, color='black', linestyle=':', label='ODF peak')  

    axs[0].set_title("w and w/o ODF")
    axs[1].set_title("w ODF")
    axs[2].set_title("w/o ODF")

    axs[-1].set_xlabel("Frequency (MHz)")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    # Plot peak shifts relative to the magnetic field
    axs[3].scatter(b_field_gauss, pos_peaks, color='r', label="Peak - Lower manifold")
    axs[3].plot(b_field_gauss, pos_peaks, color='r', linestyle="--", alpha=0.7)
    axs[3].scatter(b_field_gauss, neg_peaks, color='b', label="Peak - Upper manifold")
    axs[3].plot(b_field_gauss, neg_peaks, color='b', linestyle="--", alpha=0.7)
    axs[3].set_xlabel("Magnetic Field (Gauss)")
    axs[3].set_ylabel("Frequency (MHz)")
    axs[3].set_title("Peaks position over the magnetic field")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

    return pos_peaks, neg_peaks