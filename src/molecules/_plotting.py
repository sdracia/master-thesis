# File: _plotting.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Zeeman level plotting routines for molecular states.

Functions
---------
plot_zeeman_levels(molecule, j, title, savetext)
    Plot Zeeman energies and annotate transitions for a given j manifold.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from saving import save_figure_in_images


def plot_zeeman_levels(molecule, j: int, title: str, savetext: str):
    """
    Plot the Zeeman energies of all states in a given j manifold.

    States are colored according to spin imbalance: spin_up^2 - spin_down^2.
    Transitions between states are shown as arrows, annotated with:
        - energy difference (in kHz)
        - coupling strength (dimensionless)

    Parameters
    ----------
    molecule : object
        Molecule instance containing state_df and transition_df.
    j : int
        Rotational quantum number of the manifold to plot.
    title : str
        Title of the plot.
    savetext : str
        File name or path to save the figure via save_figure_in_images.
    """
    states_in_j = molecule.state_df.loc[molecule.state_df["j"] == j]
    transitions_in_j = molecule.transition_df[molecule.transition_df["j"] == j]

    m = states_in_j["m"].to_numpy()
    energies = states_in_j["zeeman_energy_khz"].to_numpy()
    spin_up = states_in_j["spin_up"].to_numpy()
    spin_down = states_in_j["spin_down"].to_numpy()
    colors = spin_up**2 - spin_down**2

    dim = 2 * j + 10
    fig, ax = plt.subplots(figsize=(dim, 8))

    for mi, ei, ci in zip(m, energies, colors):
        ax.hlines(ei, mi - 0.3, mi + 0.3, colors=plt.cm.plasma(ci), linewidth=5)

    ax.set_xlabel("$m_F$", fontsize=25)
    ax.set_ylabel("Zeeman energy (kHz)", fontsize=25)
    ax.set_title(title, fontsize=28)

    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap="plasma"
        ),
        ax=ax
    )
    cbar.set_label("spin $S$", fontsize=23)

    ax.set_xlim(-j - 1, j + 1)
    ax.set_xticks([i + 0.5 for i in range(-j - 1, j + 1)])

    delta = np.max(states_in_j["zeeman_energy_khz"]) - np.min(states_in_j["zeeman_energy_khz"])

    # Plot arrows for transitions and annotate energy differences and coupling
    for transition in transitions_in_j.itertuples():
        m1, xi1 = transition.m1, transition.xi1
        m2, xi2 = transition.m2, transition.xi2
        energy1 = molecule.state_df.loc[
            (molecule.state_df["j"] == j) &
            (molecule.state_df["m"] == m1) &
            (molecule.state_df["xi"] == xi1)
        ].iloc[0].zeeman_energy_khz
        energy2 = molecule.state_df.loc[
            (molecule.state_df["j"] == j) &
            (molecule.state_df["m"] == m2) &
            (molecule.state_df["xi"] == xi2)
        ].iloc[0].zeeman_energy_khz

        energy_diff, coupling = transition.energy_diff, transition.coupling

        ax.annotate(
            "",
            xy=(float(m1) - 1.0, float(energy1) + float(energy_diff)),
            xytext=(float(m1), float(energy1)),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1)
        )

        # Energy difference text
        ax.text((3 * m1 + m2) / 4.0 - 0.5, (3 * energy1 + energy2) / 4.0,
                f"{energy_diff:.2f}", fontsize=21, color="black")

        # Coupling strength text
        ax.text((3 * m1 + m2) / 4.0 - 0.5,
                (3 * energy1 + energy2) / 4.0 - (delta * 0.9 / 30) - 0.1,
                f"{coupling:.3f}", fontsize=21, color="red")

    plt.tick_params(axis='both', which='major', labelsize=25)

    save_figure_in_images(fig, savetext)
    plt.show()
    plt.close()