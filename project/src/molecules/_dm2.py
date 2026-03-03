import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from saving import save_figure_in_images


def init_transition_dm2_dataframe(self):
    """
    Initialize the transition dataframe.
    """
    transition_list = []

    for j in range(self.j_max + 1):
        """I take the molecule states at given j, and calculate molecule transitions for those, in a proper order"""
        states_in_j = self.state_df.loc[self.state_df["j"] == j]
        states_index = states_in_j.index.to_numpy()
        states_array = states_in_j.to_numpy()
        m_len = 2 * j + 1
        

        # index1 <--> index2. They are just the two states involved in the transition, not necessarily the initial and final.
        # initial and final states are defined by is_minus variable in qls.py
        for index, state1 in enumerate(states_array):
            "1 state"
            index1 = states_index[index]
            m1, xi1, zeeman_energy_khz1 = state1[1], state1[2], state1[5]

            if index == 0 or index == 1 or index == m_len:  
                """no transition from the Xi.minus left edge state"""
                continue

            if index == 2 or index == m_len + 1:
                """the 2 states right next to the Xi.minus left edge state. They only have 1 transition with the left edge state"""
                index2 = states_index[0]
                state2 = states_array[0]
                m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]

                energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, -1, -1)
                transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])
                continue
            
            """For all the other states I have two transitions: one on the same value of csi, one on the opposite value of csi"""
            """Same value of csi"""
            index2 = states_index[index - 2]
            state2 = states_array[index - 2]
            m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
            energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
            coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, -1, -1)
            transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])

            """Opposite value of csi (depending on which csi has the initial state)"""
            if xi1:  # Xi.plus
                index2 = states_index[index - m_len - 1]
                state2 = states_array[index - m_len - 1]
                m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
                energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, -1, -1)
                transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])
            else:  # Xi.minus
                index2 = states_index[index + m_len - 3]
                state2 = states_array[index + m_len - 3]
                m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
                energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, -1, -1)
                transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])

    """Transition Dataframe is built following this specific order of calculations"""
    self.transition_df = pd.DataFrame(transition_list, columns=self.transition_df_columns)


def plot_zeeman_levels_dm2(self, j: int):
    """
    Plot the Zeeman energies of all states in a given j value.
    """
    states_in_j = self.state_df.loc[self.state_df["j"] == j]
    transitions_in_j = self.transition_df[self.transition_df["j"] == j]
    m = states_in_j["m"].to_numpy()
    energies = states_in_j["zeeman_energy_khz"].to_numpy()
    spin_up = states_in_j["spin_up"].to_numpy()
    spin_down = states_in_j["spin_down"].to_numpy()
    colors = spin_up**2 - spin_down**2

    dim = 2*j+6

    fig, ax = plt.subplots(figsize=(dim, 8))
    for mi, ei, ci in zip(m, energies, colors):
        ax.hlines(ei, mi - 0.3, mi + 0.3, colors=plt.cm.plasma(ci), linewidth=3)

    ax.set_xlabel("m")
    ax.set_ylabel("Zeeman energy (kHz)")
    ax.set_title(f"Zeeman energies of all states in j={j}, B={self.b_field_gauss} G")

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap="plasma"), ax=ax)
    cbar.set_label("spin")

    ax.set_xlim(-j-1, j+1)
    ax.set_xticks([i+0.5 for i in range(-j - 1, j + 1)])

    min = np.min(states_in_j["zeeman_energy_khz"])
    max = np.max(states_in_j["zeeman_energy_khz"])

    delta = max - min


    # plot the difference between neibouring states on arrows conecting them
    for transition in transitions_in_j.itertuples():
        m1 = transition.m1
        xi1 = transition.xi1
        energy1 = self.state_df.loc[(self.state_df["j"] == j) & (self.state_df["m"] == m1) & (self.state_df["xi"] == xi1)].iloc[0].zeeman_energy_khz
        m2 = transition.m2
        xi2 = transition.xi2
        energy2 = self.state_df.loc[(self.state_df["j"] == j) & (self.state_df["m"] == m2) & (self.state_df["xi"] == xi2)].iloc[0].zeeman_energy_khz
        energy_diff = transition.energy_diff
        coupling = transition.coupling

        ax.annotate(
            "", 
            xy=(float(m1)-2.0, float(energy1) + float(energy_diff)), 
            xytext=(float(m1), float(energy1)),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1)
        )
        # add the energy difference as text on the arrow
        ax.text((3*m1 + m2) / 4.0 -0.5, (3*energy1 + energy2) / 4.0, f"{energy_diff:.2f} kHz", fontsize=10, color="black")
        # add the coupling strength as text on the arrow
        ax.text((3*m1 + m2) / 4.0 -0.5, (3*energy1 + energy2) / 4.0 - (delta*0.9/30), f"{coupling:.5f}", fontsize=10, color="red")


    plt.show()
    plt.close()
