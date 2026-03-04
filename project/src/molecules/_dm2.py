"""
DM2-specific transition and Zeeman level routines.

Functions
---------
init_transition_dm2_dataframe(self)
    Initialize the transition dataframe for DM2 molecule states.

plot_zeeman_levels_dm2(self, j)
    Plot Zeeman energies of all states in a given j manifold,
    including arrows indicating transitions and annotated coupling strengths.
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def init_transition_dm2_dataframe(self):
    """
    Initialize the DM2 transition dataframe.

    Each state generates two transitions:
        1. within the same Xi manifold
        2. to the opposite Xi manifold
    except for edge states which may only have one transition.

    The result is stored in self.transition_df with columns:
        ["j", "m1", "xi1", "m2", "xi2", "index1", "index2", "energy_diff", "coupling"]
    """
    transition_list = []

    for j in range(self.j_max + 1):
        states_in_j = self.state_df.loc[self.state_df["j"] == j]
        states_index = states_in_j.index.to_numpy()
        states_array = states_in_j.to_numpy()
        m_len = 2 * j + 1

        # index1 <--> index2: indices of the two states involved in the transition
        # initial and final states are defined by is_minus variable in qls.py
        for index, state1 in enumerate(states_array):
            index1 = states_index[index]
            m1, xi1, zeeman_energy_khz1 = state1[1], state1[2], state1[5]

            if index == 0 or index == 1 or index == m_len:
                # no transition from the Xi.minus left edge state
                continue

            if index == 2 or index == m_len + 1:
                # the 2 states right next to the Xi.minus left edge state; only one transition
                index2 = states_index[0]
                state2 = states_array[0]
                m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]

                energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, -1, -1)
                transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])
                continue

            # Two transitions for other states: same Xi and opposite Xi
            # Same Xi
            index2 = states_index[index - 2]
            state2 = states_array[index - 2]
            m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
            energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
            coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, -1, -1)
            transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])

            # Opposite Xi
            if xi1:  # Xi.plus
                index2 = states_index[index - m_len - 1]
                state2 = states_array[index - m_len - 1]
            else:  # Xi.minus
                index2 = states_index[index + m_len - 3]
                state2 = states_array[index + m_len - 3]

            m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
            energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
            coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, -1, -1)
            transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])

    # Build transition dataframe
    self.transition_df = pd.DataFrame(transition_list, columns=self.transition_df_columns)


def plot_zeeman_levels_dm2(self, j: int):
    """
    Plot the Zeeman energies of all states in a given j manifold.

    States are colored by spin imbalance (spin_up^2 - spin_down^2).
    Arrows indicate transitions with annotated energy differences (kHz)
    and coupling strengths.

    Parameters
    ----------
    j : int
        Rotational quantum number of the manifold to plot.
    """
    states_in_j = self.state_df.loc[self.state_df["j"] == j]
    transitions_in_j = self.transition_df[self.transition_df["j"] == j]

    m = states_in_j["m"].to_numpy()
    energies = states_in_j["zeeman_energy_khz"].to_numpy()
    spin_up = states_in_j["spin_up"].to_numpy()
    spin_down = states_in_j["spin_down"].to_numpy()
    colors = spin_up**2 - spin_down**2

    dim = 2 * j + 6
    fig, ax = plt.subplots(figsize=(dim, 8))

    for mi, ei, ci in zip(m, energies, colors):
        ax.hlines(ei, mi - 0.3, mi + 0.3, colors=plt.cm.plasma(ci), linewidth=3)

    ax.set_xlabel("m")
    ax.set_ylabel("Zeeman energy (kHz)")
    ax.set_title(f"Zeeman energies of all states in j={j}, B={self.b_field_gauss} G")

    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap="plasma"),
        ax=ax
    )
    cbar.set_label("spin")

    ax.set_xlim(-j - 1, j + 1)
    ax.set_xticks([i + 0.5 for i in range(-j - 1, j + 1)])

    delta = np.max(states_in_j["zeeman_energy_khz"]) - np.min(states_in_j["zeeman_energy_khz"])

    # Plot arrows and annotate transitions
    for transition in transitions_in_j.itertuples():
        m1, xi1 = transition.m1, transition.xi1
        m2, xi2 = transition.m2, transition.xi2
        energy1 = self.state_df.loc[(self.state_df["j"] == j) & (self.state_df["m"] == m1) & (self.state_df["xi"] == xi1)].iloc[0].zeeman_energy_khz
        energy2 = self.state_df.loc[(self.state_df["j"] == j) & (self.state_df["m"] == m2) & (self.state_df["xi"] == xi2)].iloc[0].zeeman_energy_khz
        energy_diff, coupling = transition.energy_diff, transition.coupling

        ax.annotate(
            "",
            xy=(float(m1) - 2.0, float(energy1) + float(energy_diff)),
            xytext=(float(m1), float(energy1)),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1)
        )

        ax.text((3 * m1 + m2) / 4.0 - 0.5, (3 * energy1 + energy2) / 4.0,
                f"{energy_diff:.2f} kHz", fontsize=10, color="black")
        ax.text((3 * m1 + m2) / 4.0 - 0.5, (3 * energy1 + energy2) / 4.0 - (delta * 0.9 / 30),
                f"{coupling:.5f}", fontsize=10, color="red")

    plt.show()
    plt.close()