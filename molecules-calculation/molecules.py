import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from typing import TypeAlias
from pathlib import Path
from wigners import clebsch_gordan
from scipy.constants import h, physical_constants
import qls

matplotlib.use("TkAgg")

# physical constants
mu_N = physical_constants["nuclear magneton"][0]
gI = physical_constants["proton g factor"][0]


class CaH:
    name: str = "CaH"
    """name of the molecule"""
    gj: float = -1.36
    """g factor for J"""
    cij_khz: float = 8.52
    """coupling strength between proton spin and molecule rotation, in kHz"""
    br_ghz: float = 144.0
    """rotational constant, in Hz"""
    omega_0_thz: float = 750.0
    """frequency of the electronic transition between ground state and 1st excited state, in THz"""
    omega_thz: float = 285.5
    """frequency of the Raman beam, in THz"""
    coupling_coefficient: float = 1.
    """coupling coefficient, for now to be able to compare with the NIST value"""

    def __init__(self, b_field_gauss: float, j_max: int) -> None:
        self.b_field_gauss = b_field_gauss
        """magnetic field in Gauss"""

        self.j_max = j_max
        """maximum j value to consider"""

        self.cb_khz = mu_N * b_field_gauss * 1e-4 / h / 1e3
        """zeeman coefficient mu_N * B / h"""

        self.state_df: pd.DataFrame = pd.DataFrame()
        self.state_df_columns = ["j", "m", "xi", "spin_up", "spin_down", "zeeman_energy_khz", "rotation_energy_ghz"]
        """
        j: int, the j value of the state
        m: int, the m value of the state
        xi: Xi, the xi value of the state, boolean, False for Xi.minus and True for Xi.plus
        spin_up: float, the component for state with nuclear spin aligned to rotation
        spin_down: float, the component for state with nuclear spin anti-aligned to rotation
        zeeman_energy_khz: float, the Zeeman energy in kHz
        rotation_energy_ghz: float, the rotational energy in GHz
        """

        self.transition_df: pd.DataFrame = pd.DataFrame()
        self.transition_df_columns = ["j", "m1", "xi1", "m2", "xi2", "index1", "index2", "energy_diff", "coupling"]
        """
        j: int, the j value of the state
        m1: int, the m value of the initial state
        xi1: Xi, the xi value of the initial state, boolean, False for Xi.minus and True for Xi.plus
        m2: int, the m value of the final state
        xi2: Xi, the xi value of the final state, boolean, False for Xi.minus and True for Xi.plus
        energy_diff: float, the energy difference between the final and initial state in kHz
        coupling: float, the coupling strength between the initial and final state
        """

    @classmethod
    def from_file(cls, b_field_gauss: float, j_max: int):
        """
        Load the molecule data from the file. If the file does not exist, calculate the data from the scratch.
        """
        new_instance = cls(b_field_gauss, j_max)
        states_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_states.csv")
        transitions_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_transitions.csv")
        if states_file.exists() and transitions_file.exists():
            new_instance.state_df = pd.read_csv(states_file)
            new_instance.transition_df = pd.read_csv(transitions_file)
        else:
            new_instance.init_states()
            new_instance.init_transition_dataframe()
            new_instance.save_data()
        return new_instance

    @classmethod
    def from_calculation(cls, b_field_gauss: float, j_max: int):
        """
        Calculate the molecule data from the scratch.
        """
        new_instance = cls(b_field_gauss, j_max)
        new_instance.init_states()
        new_instance.init_transition_dataframe()
        new_instance.save_data()
        return new_instance

    @classmethod
    def get_m_minus_in_j(cls, j: int) -> np.array:
        """
        j = 4
        np.arange(-j - 0.5, j, 1) --> [-4.5 -3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5]
        """
        return np.arange(-j - 0.5, j, 1)

    @classmethod
    def get_m_plus_in_j(cls, j: int) -> np.array:
        """
        j = 4
        np.arange(-j - 0.5, j, 1) --> [-3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5]
        """
        return np.arange(-j + 0.5, j + 1, 1)

    def init_states(self):
        """
        Initialize the states in a given j value.
        Calculate the Zeeman energy and the state vector for each state.
        The results are stored in the state_df as a Pandas dataframe.
        States are listed in the order of J values, starting from the lowest J to J_max.
        For each J, first Xi.minus and then Xi.plus states.
        The states of each Xi manifold are listed in the order of m values, starting from the lowest m to the highest m.
        """
        for j in range(self.j_max + 1):
            rotation_energy_ghz = self.br_ghz * j * (j + 1)
            state_list = []

            xi = False  # calculate Xi.minus states
            for m in self.get_m_minus_in_j(j):
                x = 1 / 2 * sqrt(self.cij_khz**2 * ((j + 1 / 2) ** 2 - m**2) + (self.cij_khz * m - self.cb_khz * (self.gj - gI)) ** 2)
                y = -self.cb_khz / 2 * (self.gj - gI) + m * self.cij_khz / 2
                if m == -j - 0.5:
                    spin_up = 0.0
                    spin_down = 1.0
                    # zeeman_energy_khz = (self.gj * j + gI / 2) * self.cb_khz - self.cij_khz * j / 2
                    zeeman_energy_khz = (self.gj * j + gI / 2) * self.cb_khz + self.cij_khz * j / 2
                    state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])
                else:
                    spin_up = sqrt((x - y) / (2 * x))
                    spin_down = -sqrt((x + y) / (2 * x))
                    zeeman_energy_khz = self.cij_khz / 4 - self.cb_khz * self.gj * m + x
                    state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])

            xi = True  # calculate Xi.plus states
            for m in self.get_m_plus_in_j(j):
                x = 1 / 2 * sqrt(self.cij_khz**2 * ((j + 1 / 2) ** 2 - m**2) + (self.cij_khz * m - self.cb_khz * (self.gj - gI)) ** 2)
                y = -self.cb_khz / 2 * (self.gj - gI) + m * self.cij_khz / 2
                if m == j + 0.5:
                    spin_up = 1.0
                    spin_down = 0.0
                    zeeman_energy_khz = -(self.gj * j + gI / 2) * self.cb_khz - self.cij_khz * j / 2
                    state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])
                else:
                    spin_up = sqrt((x + y) / (2 * x))
                    spin_down = sqrt((x - y) / (2 * x))
                    zeeman_energy_khz = self.cij_khz / 4 - self.cb_khz * self.gj * m - x
                    state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])

            self.state_df = pd.concat([self.state_df, pd.DataFrame(state_list, columns=self.state_df_columns)], ignore_index=True)

    def init_transition_dataframe(self):
        """
        Initialize the transition dataframe.
        """
        for j in range(self.j_max + 1):
            states_in_j = self.state_df.loc[self.state_df["j"] == j]
            states_index = states_in_j.index.to_numpy()
            states_array = states_in_j.to_numpy()
            m_len = 2 * j + 1
            transition_list = []

            # index1 --> index2
            for index, state1 in enumerate(states_array):
                index1 = states_index[index]
                m1, xi1, zeeman_energy_khz1 = state1[1], state1[2], state1[5]

                if index == 0:  # no transition from the Xi.minus left edge state
                    continue

                if index == 1 or index == m_len:
                    # the states right next to the Xi.minus left edge state
                    index2 = states_index[0]
                    state2 = states_array[0]
                    m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]

                    energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                    coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, 0, -1)
                    transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])
                    continue

                index2 = states_index[index - 1]
                state2 = states_array[index - 1]
                m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
                energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, 0, -1)
                transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])

                if xi1:  # Xi.plus
                    index2 = states_index[index - m_len]
                    state2 = states_array[index - m_len]
                    m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
                    energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                    coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, 0, -1)
                    transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])
                else:  # Xi.minus
                    index2 = states_index[index + m_len - 2]
                    state2 = states_array[index + m_len - 2]
                    m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
                    energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                    coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, 0, -1)
                    transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])

            self.transition_df = pd.concat([self.transition_df, pd.DataFrame(transition_list, columns=self.transition_df_columns)], ignore_index=True)

    def get_raman_coupling(self, index1, index2, qa, qb):
        """
        Calculates the coupling between two Zeeman levels, |m1,xi1> & |m2,xi2>
        with two Raman beam with polarization qa and qb.
        Detuning and the electronic transition term is not considered.
        """

        def j_coupling(j, j_exc, mj1, mj2, qa, qb) -> float:
            if j < 0 or j_exc < 0:
                return 0
            if j < abs(mj1) or j < abs(mj2) or j_exc < abs(mj1 + qa):
                return 0
            return (
                np.sqrt((2 * j_exc + 1) / (2 * j + 1))
                * clebsch_gordan(1, 0, j_exc, 0, j, 0)
                * clebsch_gordan(1, -qa, j_exc, mj1 + qa, j, mj1)
                * np.sqrt((2 * j + 1) / (2 * j_exc + 1))
                * clebsch_gordan(1, 0, j, 0, j_exc, 0)
                * clebsch_gordan(1, -qb, j, mj2, j_exc, mj1 + qa)
            )

        state1 = self.state_df.loc[index1]
        state2 = self.state_df.loc[index2]
        j = state1.j
        m1 = state1.m
        m2 = state2.m
        m1_up = int(m1 - 0.5)
        m1_down = int(m1 + 0.5)
        m2_up = int(m2 - 0.5)
        m2_down = int(m2 + 0.5)

        coupling_minus = (
            1.0
            / (self.omega_thz - self.omega_0_thz)
            * (
                state1.spin_down * state2.spin_down * j_coupling(j, j + 1, m1_down, m2_down, qa, qb)
                + state1.spin_down * state2.spin_up * j_coupling(j, j + 1, m1_down, m2_up, qa, qb)
                + state1.spin_up * state2.spin_down * j_coupling(j, j + 1, m1_up, m2_down, qa, qb)
                + state1.spin_up * state2.spin_up * j_coupling(j, j + 1, m1_up, m2_up, qa, qb)
                + state1.spin_down * state2.spin_down * j_coupling(j, j - 1, m1_down, m2_down, qa, qb)
                + state1.spin_down * state2.spin_up * j_coupling(j, j - 1, m1_down, m2_up, qa, qb)
                + state1.spin_up * state2.spin_down * j_coupling(j, j - 1, m1_up, m2_down, qa, qb)
                + state1.spin_up * state2.spin_up * j_coupling(j, j - 1, m1_up, m2_up, qa, qb)
            )
        )

        coupling_plus = (
            1.0
            / (self.omega_0_thz + self.omega_thz)
            * (
                state1.spin_down * state2.spin_down * j_coupling(j, j + 1, m1_down, m2_down, qb, qa)
                + state1.spin_down * state2.spin_up * j_coupling(j, j + 1, m1_down, m2_up, qb, qa)
                + state1.spin_up * state2.spin_down * j_coupling(j, j + 1, m1_up, m2_down, qb, qa)
                + state1.spin_up * state2.spin_up * j_coupling(j, j + 1, m1_up, m2_up, qb, qa)
                + state1.spin_down * state2.spin_down * j_coupling(j, j - 1, m1_down, m2_down, qb, qa)
                + state1.spin_down * state2.spin_up * j_coupling(j, j - 1, m1_down, m2_up, qb, qa)
                + state1.spin_up * state2.spin_down * j_coupling(j, j - 1, m1_up, m2_down, qb, qa)
                + state1.spin_up * state2.spin_up * j_coupling(j, j - 1, m1_up, m2_up, qb, qa)
            )
        )

        return (coupling_minus + coupling_plus) / (1.0 / (self.omega_thz - self.omega_0_thz) + 1.0 / (self.omega_0_thz + self.omega_thz))

    def plot_zeeman_levels(self, j: int):
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

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(m, energies, marker="_", c=colors, cmap="plasma", s=500, linewidths=5)
        ax.set_xlabel("m")
        ax.set_ylabel("Zeeman energy (kHz)")
        ax.set_title(f"Zeeman energies of all states in j={j}, B={self.b_field_gauss} G")
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap="plasma"), ax=ax)

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
            ax.arrow(float(m1), float(energy1), -1.0, float(energy_diff), head_width=0.1, head_length=0.1, linestyle="dotted", color="black")
            # add the energy difference as text on the arrow
            ax.text((m1 + m2) / 2.0 - 0.2, (energy1 + energy2) / 2.0, f"{energy_diff:.3f}", fontsize=10)
            # add the coupling strength as text on the arrow
            ax.text((m1 + m2) / 2.0, (energy1 + energy2) / 2.0 - 0.9, f"{coupling:.3f}", fontsize=10, color="red")

        plt.show()
        plt.close()

    def prova(self, j: int, data_ac: list):
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


        # With AC stark shift
        rabi_rate_mhz_1 = data_ac[0] # 0.002
        rabi_rate_mhz_2 = data_ac[1] # 0.004
        q1 = data_ac[2] # qls.Polarization(1, 0, 0)
        q2 = data_ac[3] # qls.Polarization(0, 0, 1)
        ac_stark_shifts = qls.get_ac_stark_shifts(self, rabi_rate_mhz_1, rabi_rate_mhz_2, q1, q2)
        energies_shifted = energies + ac_stark_shifts[states_in_j.index] * 1e3

        diff = energies_shifted - energies


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        ax1.scatter(m, energies, marker="_", c=colors, cmap="plasma", s=500, linewidths=5)
        ax1.scatter(m, energies_shifted, marker=".", c=colors, cmap="plasma", s=500, linewidths=5)
        ax1.set_xlabel("m")
        ax1.set_ylabel("Zeeman energy (kHz)")
        ax1.set_title(f"Zeeman energies of all states in j={j}, B={self.b_field_gauss} G")
        ax1.set_xticks(m)

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
            ax1.arrow(float(m1), float(energy1), -1.0, float(energy_diff), head_width=0.1, head_length=0.1, linestyle="dotted", color="black")
            # add the energy difference as text on the arrow
            ax1.text((m1 + m2) / 2.0 - 0.2, (energy1 + energy2) / 2.0, f"{energy_diff:.3f}", fontsize=10)
            # add the coupling strength as text on the arrow
            ax1.text((m1 + m2) / 2.0, (energy1 + energy2) / 2.0 - 0.9, f"{coupling:.3f}", fontsize=10, color="red")


        ax2.scatter(m, diff, c=colors)
        ax2.set_xlabel("m")
        ax2.set_ylabel("Energy difference (kHz)")
        ax2.set_title(f"Energy difference: comparison WITH and WITHOUT AC Stark shift. j={j}, B={self.b_field_gauss} G")
        ax2.set_xticks(m)
        ax2.grid()

        fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap="plasma"), ax=[ax1])
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap="plasma"), ax=[ax2])

        plt.show()
        plt.close()

    def save_data(self):
        self.state_df.to_csv(f"molecule_data/{self.name}_B[{self.b_field_gauss:.2f}]_Jmax[{self.j_max}]_states.csv", index=False)
        self.transition_df.to_csv(f"molecule_data/{self.name}_B[{self.b_field_gauss:.2f}]_Jmax[{self.j_max}]_transitions.csv", index=False)


class CaOH(CaH):
    name: str = "CaOH"
    """name of the molecule"""
    gj: float = -0.036
    """g factor for J"""
    cij_khz: float = 1.49
    """coupling strength between proton spin and molecule rotation, in kHz"""
    br_ghz: float = 11.0
    """rotational constant, in GHz"""
    omega_0_thz: float = 1100.0
    """frequency of the electronic transition between ground state and 1st excited state, in THz"""
    omega_thz: float = 280.0
    """frequency of the Raman beam, in THz"""
    coupling_coefficient: float = 1.
    """coupling coefficient, for now to be able to compare with the NIST value"""

    def __init__(self, b_field_gauss: float, j_max: int) -> None:
        super().__init__(b_field_gauss, j_max)


class CaOH_dm2(CaH):
    name: str = "CaOH_dm2"
    """name of the molecule"""
    gj: float = -0.036
    """g factor for J"""
    cij_khz: float = 1.49
    """coupling strength between proton spin and molecule rotation, in kHz"""
    br_ghz: float = 11.0
    """rotational constant, in GHz"""
    omega_0_thz: float = 1100.0
    """frequency of the electronic transition between ground state and 1st excited state, in THz"""
    omega_thz: float = 280.0
    """frequency of the Raman beam, in THz"""
    coupling_coefficient: float = 1.0
    """coupling coefficient, for now to be able to compare with the NIST value"""

    def __init__(self, b_field_gauss: float, j_max: int) -> None:
        super().__init__(b_field_gauss, j_max)

    @classmethod
    def from_file_dm2(cls, b_field_gauss: float, j_max: int):
        """
        Load the molecule data from the file. If the file does not exist, calculate the data from the scratch.
        """
        new_instance = cls(b_field_gauss, j_max)
        states_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_states.csv")
        transitions_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_transitions.csv")
        if states_file.exists() and transitions_file.exists():
            new_instance.state_df = pd.read_csv(states_file)
            new_instance.transition_df = pd.read_csv(transitions_file)
        else:
            new_instance.init_states()
            new_instance.init_transition_dm2_dataframe()
            new_instance.save_data()
        return new_instance

    @classmethod
    def from_calculation_dm2(cls, b_field_gauss: float, j_max: int):
        """
        Calculate the molecule data from the scratch.
        """
        new_instance = cls(b_field_gauss, j_max)
        new_instance.init_states()
        new_instance.init_transition_dm2_dataframe()
        new_instance.save_data()
        return new_instance

    def init_transition_dm2_dataframe(self):
        """
        Initialize the transition dataframe.
        """
        for j in range(self.j_max + 1):
            states_in_j = self.state_df.loc[self.state_df["j"] == j]
            states_index = states_in_j.index.to_numpy()
            states_array = states_in_j.to_numpy()
            m_len = 2 * j + 1
            transition_list = []
            for index, state1 in enumerate(states_array):
                index1 = states_index[index]
                m1, xi1, zeeman_energy_khz1 = state1[1], state1[2], state1[5]

                if index == 0 or index == 1 or index == m_len:
                    # no transition from the Xi.minus left two edge state
                    # no transition from the Xi.plus left edge state
                    continue

                if index == 2 or index == m_len + 1:
                    # the states right next to the Xi.minus left edge state
                    index2 = states_index[0]
                    state2 = states_array[0]
                    m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]

                    energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                    coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, -1, -1)
                    transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])
                    continue

                index2 = states_index[index - 2]
                state2 = states_array[index - 2]
                m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
                energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, -1, -1)
                transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])

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

            self.transition_df = pd.concat([self.transition_df, pd.DataFrame(transition_list, columns=self.transition_df_columns)], ignore_index=True)

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

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(m, energies, marker="_", c=colors, cmap="plasma", s=500, linewidths=5)
        ax.set_xlabel("m")
        ax.set_ylabel("Zeeman energy (kHz)")
        ax.set_title(f"Zeeman energies of all states in j={j}, B={self.b_field_gauss} G")
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap="plasma"), ax=ax)

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
            ax.arrow(float(m1), float(energy1), -2.0, float(energy_diff), head_width=0.1, head_length=0.1, linestyle="dotted", color="black")
            # add the energy difference as text on the arrow
            ax.text((m1 + m2) / 2.0 - 0.2, (energy1 + energy2) / 2.0, f"{energy_diff:.3f}", fontsize=10)
            # add the coupling strength as text on the arrow
            ax.text((m1 + m2) / 2.0, (energy1 + energy2) / 2.0 - 0.9, f"{coupling:.3f}", fontsize=10, color="red")

        plt.show()
        plt.close()


Molecule: TypeAlias = CaH | CaOH | CaOH_dm2

if __name__ == "__main__":
    # test = CaH(6.5)
    # J = 6
    # test.plot_zeeman_levels(J, include_j_to_j=False)
    test2 = CaOH(3.1, 30)
    J = 2
    # test2.init_transition_dataframe(J)
    # J = 1
    # test2.init_transition_dataframe(J)
    # print(test2.transition_df)
    # test2.plot_zeeman_levels(J)
