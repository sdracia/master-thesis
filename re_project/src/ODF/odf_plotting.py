from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from saving import save_figure_in_images
from utils import find_main_peaks
from QLS.spectrum import get_spectrum
from QLS.state_dist import *
from molecules.molecule import CaOH
from ODF.data_fitting import sinc


###################################################################################
########################## MOLECULE + MOTION + ION ################################
########################## MOLECULE WITH 3 LEVELS  ################################
###################################################################################

def plot_atom_mol3_evolution(x_data, y_data, fin_time, w_mol, rabi_rate_molecule, rabi_rate, detunings, exc_mol_odf, exc_mol_bsb, exc_mol_tot, spin_up_atom_odf, spin_up_atom_bsb, spin_up_atom_tot, B=None):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # --- Subplot 1: Molecule ---
    ax[0].plot(detunings/(2*np.pi*1e-3), exc_mol_bsb, color='black', label='BSB', linestyle='-', linewidth=3)
    ax[0].plot(detunings/(2*np.pi*1e-3), exc_mol_odf, color='black', label='ODF', linestyle=':', linewidth=3)
    ax[0].plot(detunings/(2*np.pi*1e-3), exc_mol_tot, color='red', label='BSB+ODF', linestyle='--', linewidth=3)
    ax[0].set_xlabel("Detuning (kHz)", fontsize=25)
    ax[0].set_ylabel(r"$\langle n \rangle$", fontsize=25)
    ax[0].set_title("Average phonon number ", fontsize=28)
    # ax[0].axvline(x=w_mol, color='black', linestyle='--', linewidth=2.5)
    ax[0].legend(frameon=True)
    ax[0].grid(True, which='major', linestyle='--', alpha=0.5)
    ax[0].tick_params(axis='both', which='major', labelsize=25)
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=5))

    # --- Subplot 2: ODF experimental data ---
    ax[1].plot(detunings/(2*np.pi*1e-3), spin_up_atom_odf, color='black', label='ODF', linestyle=':', linewidth=3)
    ax[1].scatter(x_data/(1e-3), y_data, color='red', label='Data')  # triangoli rossi
    # ax[1].plot(freq, y_estimated, color='blue', label='Estimation')
    ax[1].set_xlabel("Detuning (kHz)", fontsize=25)
    ax[1].set_ylabel("Excitation probability", fontsize=25)
    ax[1].set_title("ODF experimental data", fontsize=28)
    # ax[1].axvline(x=w_mol, color='black', linestyle='--', linewidth=2.5)
    ax[1].legend(frameon=True)
    # ax[1].grid(True)
    ax[1].grid(True, which='major', linestyle='--', alpha=0.5)
    ax[1].tick_params(axis='both', which='major', labelsize=25)
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=5))


    # --- Subplot 3: RSB readout ---
    ax[2].plot(detunings/(2*np.pi*1e-3), spin_up_atom_bsb, color='black', label='BSB', linestyle='-', linewidth=3)
    ax[2].plot(detunings/(2*np.pi*1e-3), spin_up_atom_odf, color='black', label='ODF', linestyle=':', linewidth=3)
    ax[2].plot(detunings/(2*np.pi*1e-3), spin_up_atom_tot, color='red', label='BSB+ODF', linestyle='--', linewidth=3)
    ax[2].set_xlabel("Detuning (kHz)", fontsize=25)
    ax[2].set_ylabel("Excitation probability", fontsize=25)
    ax[2].set_title("RSB readout", fontsize=28)
    # ax[2].axvline(x=w_mol, color='black', linestyle='--', linewidth=2.5)
    ax[2].legend(frameon=True)
    # ax[2].grid(True)
    ax[2].grid(True, which='major', linestyle='--', alpha=0.5)

    ax[2].tick_params(axis='both', which='major', labelsize=25)
    ax[2].xaxis.set_major_locator(MaxNLocator(nbins=5))

    # --- Super title ---
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


def plot_odf_3mol_data(detunings, spin_up_atom_odf, x_data, y_data):

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(detunings/(2*np.pi*1e-3), spin_up_atom_odf, color='black', label='Simulated ODF', linestyle=':', linewidth=3)
    # ax.scatter(x_data/(1e-3), y_data, color='red', label='Data')  # triangoli rossi
    ax.plot(
        x_data/(1e-3),
        y_data,
        color='red',
        marker='^',     # triangoli
        markersize=10,  # dimensione dei marker
        linewidth=2,    # spessore della linea
        label='Data'
    )
    # ax.plot(freq, y_estimated, color='blue', label='Estimation')
    ax.set_xlabel("Detuning (kHz)", fontsize=25)
    ax.set_ylabel("Excitation probability", fontsize=25)
    ax.set_title("ODF experimental data", fontsize=28)
    # ax.axvline(x=w_mol, color='black', linestyle='--', linewidth=2.5)
    ax.legend(frameon=True, fontsize =25)
    # ax.grid(True)
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout()
    save_figure_in_images(fig, "ODF_experimental_data.pdf")
    plt.show()








###################################################################################
########################## MOLECULE + MOTION + ION ################################
###################################################################################


def plot_atom_mol_evolution(w_mol, rabi_rate_molecule, rabi_rate, detunings, exc_mol_odf, exc_mol_bsb, exc_mol_tot, spin_up_mol_odf, spin_up_mol_bsb, spin_up_mol_tot, spin_up_atom_odf, spin_up_atom_bsb, spin_up_atom_tot):
    plt.figure(figsize=(18, 4))

    plt.subplot(1,3,1)
    plt.plot(detunings/(2*np.pi), exc_mol_bsb, color='black', label='BSB')
    plt.plot(detunings/(2*np.pi), exc_mol_odf, color='black', label='ODF', linestyle=':')
    plt.plot(detunings/(2*np.pi), exc_mol_tot, color='red', label='BSB+ODF', linestyle='--')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("<n>")
    plt.title(f"MOLECULE <n>")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()

    plt.subplot(1,3,2)
    plt.plot(detunings/(2*np.pi), spin_up_mol_bsb, color='black', label='BSB')
    plt.plot(detunings/(2*np.pi), spin_up_mol_odf, color='black', label='ODF', linestyle=':')
    plt.plot(detunings/(2*np.pi), spin_up_mol_tot, color='red', label='BSB+ODF', linestyle='--')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("|<spin_up>|^2")
    plt.title(f"MOLECULE spin_up")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()

    plt.subplot(1,3,3)
    plt.plot(detunings/(2*np.pi), spin_up_atom_bsb, color='black', label='BSB')
    plt.plot(detunings/(2*np.pi), spin_up_atom_odf, color='black', label='ODF', linestyle=':')
    plt.plot(detunings/(2*np.pi), spin_up_atom_tot, color='red', label='BSB+ODF', linestyle='--')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("|<spin_up>|^2")
    plt.title(f"RSB readout with pi-time")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()

    plt.suptitle(f"|spin_down>|0>|spin_down>. mol trans = {w_mol*1e3}kHz. RR_mol={rabi_rate_molecule:.6f}; RR_odf={rabi_rate:.6f}", fontsize=12)
    plt.tight_layout()

    plt.show()





###################################################################################
########################## MOLECULE + MOTION ######################################
###################################################################################


def odf_peak_phase_space(x_vals, p_vals, detunings, exc):
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(detunings/(2*np.pi), exc, marker='o', linestyle='-', color='blue')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("<n>")
    plt.title("ODF Peak Scan")
    # plt.ylim(0,0.007)

    plt.subplot(1, 2, 2)
    plt.scatter(x_vals, p_vals, c=detunings, cmap='viridis', edgecolors='k')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.colorbar(label="Detuning (MHz)")
    plt.xlabel("X quadrature (<x>)")
    plt.ylabel("P quadrature (<p>)")
    plt.title("Final Phase space state")

    plt.tight_layout()
    plt.show()



def plot_mol_evolution_phase_space(detunings, exc, prob, spin_up, x_vals, p_vals, w_mol):
    plt.figure(figsize=(18, 5))


    plt.subplot(1, 3, 1)
    plt.plot(detunings/(2*np.pi), exc, marker='o', linestyle='-', label="BSB+ODF - <n>", color='blue')
    plt.plot(detunings/(2*np.pi), spin_up, label="BSB+ODF - |<spin_up>|^2", color = 'red', linestyle = "-")
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("average values")
    plt.title(f"|spin_down>|0>, mol trans = {w_mol*1e3}kHz")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1, label=f"{w_mol*1e3}kHz")
    plt.legend(loc="upper right", fontsize=12)
    # plt.xlim(-0.002, 0.002)
    # plt.ylim(0,0.05)


    plt.subplot(1, 3, 2)
    plt.plot(detunings/(2*np.pi), prob, marker='o', linestyle='-', label="BSB+ODF - 1-|<0|0>|^2", color = "blue")
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("excitation probability")
    plt.title(f"|spin_down>|0>, mol trans = {w_mol*1e3}kHz")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1, label=f"{w_mol*1e3}kHz")
    plt.legend(loc="upper right", fontsize=12)
    # plt.xlim(-0.002, 0.002)
    # plt.ylim(0,0.05)

    plt.subplot(1, 3, 3)
    plt.scatter(x_vals, p_vals, c=detunings, cmap='viridis', edgecolors='k')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.colorbar(label="Detuning (MHz)")
    plt.xlabel("X quadrature (<x>)")
    plt.ylabel("P quadrature (<p>)")
    plt.title("Final Phase space state")

    plt.tight_layout()
    plt.show()



def odf_peak_comparison(exc_bsb, exc_tot, spin_up_bsb, spin_up_tot, detunings, w_mol):
    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    plt.plot(detunings/(2*np.pi), exc_bsb, color='black', label='BSB', linestyle='-')
    plt.plot(detunings/(2*np.pi), exc_tot, color='red', label='BSB+ODF', linestyle='--')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("<n>")
    plt.title(f"|spin_down>|0>, mol trans = {w_mol*1e3}kHz \n <n>")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(detunings/(2*np.pi), spin_up_bsb, color='black', label='BSB', linestyle='-')
    plt.plot(detunings/(2*np.pi), spin_up_tot, color='red', label='BSB+ODF', linestyle='--')
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("|<spin_up>|^2")
    plt.title(f"|spin_down>|0>, mol trans = {w_mol*1e3}kHz \n spin_up")
    plt.axvline(x=w_mol, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


    plt.style.use(['science', 'notebook', 'ieee'])
    plt.rcParams['font.family'] = 'Times New Roman'




###################################################################



def classical_odf_varying_B(b_field_gauss, j_max, temperature, spectrum_list, result):

    plt.style.use('default')


    pos_peaks = []
    neg_peaks = []

    cmap = plt.get_cmap("plasma") 
    norm = plt.Normalize(vmin=min(b_field_gauss), vmax=max(b_field_gauss))
    colors = [cmap(norm(b)) for b in b_field_gauss]


    fig, axs = plt.subplots(4, 1, figsize=(12, 18))

    for b, color in zip(b_field_gauss, colors):
        mo1 = CaOH.create_molecule_data(b_field_gauss=b, j_max=j_max)
        states1 = States(mo1, temperature)

        frequencies, exc_probs = get_spectrum(
            molecule=mo1,
            state_distribution=mo1.state_df["state_dist"],
            **spectrum_list,
            noise_params=None,
            seed=None
        )

        odf_false_positive = sinc(frequencies, **result.best_values)

        exc_probs_odf = exc_probs + odf_false_positive
        freq_neg_peak, freq_pos_peak = find_main_peaks(frequencies, exc_probs)
        neg_peaks.append(freq_neg_peak)
        pos_peaks.append(freq_pos_peak)

        axs[0].plot(frequencies, exc_probs_odf, color=color, linestyle='-', label=f"w/ODF; B={b} G")
        axs[0].plot(frequencies, exc_probs, color=color, linestyle='--') 

        axs[1].plot(frequencies, exc_probs_odf, color=color, linestyle='-', label=f"w/ODF; B={b} G")

        axs[2].plot(frequencies, exc_probs, color=color, linestyle='--', label=f"w/ODF; B={b} G")


    for ax in axs[0:3]:
        ax.set_xlim([-spectrum_list['max_frequency_mhz'], spectrum_list['max_frequency_mhz']])
        ax.set_xlim(-0.004, 0.004)
        ax.set_ylabel("Excitation probability")

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
    # axs[2].set_xlim(-0.004, 0.004)

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