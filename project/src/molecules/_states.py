import numpy as np
import pandas as pd
from math import sqrt
from scipy.constants import physical_constants


gI = physical_constants["proton g factor"][0]


def m_csi_minus(cls, j: int) -> np.array:
    """
    Computes the values of m, for a given J, with csi = -

    j = 4
    m_csi_minus(j) --> [-4.5 -3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5]
    """
    return np.arange(-j - 0.5, j, 1)


def m_csi_plus(cls, j: int) -> np.array:
    """
    Computes the values of m, for a given J, with csi = +

    j = 4
    m_csi_minus(j) --> [-3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5]
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

    state_list = []

    for i,j in enumerate(range(self.j_max + 1)):

        gj = self.gj_list[i] if self.gj_list != [] else self.gj
        cij = self.cij_list[i] if self.gj_list != [] else self.cij_khz
        """If the lists exist, they are taken as gj and cij for different j's, otherwise the single value is considered"""


        rotation_energy_ghz = self.br_ghz * j * (j + 1)
        
        zeeman_edge_minus = (gj * j + gI / 2) * self.cb_khz - cij * j / 2
        zeeman_edge_plus = -(gj * j + gI / 2) * self.cb_khz - cij * j / 2
        """j-independent energies for the edge states"""

        xi = False  # calculate xi = - states
        for m in self.m_csi_minus(j):
            x = 1 / 2 * sqrt(cij**2 * ((j + 1 / 2) ** 2 - m**2) + (cij * m - self.cb_khz * (gj - gI)) ** 2)
            y = -self.cb_khz * (gj - gI) / 2  + m * cij / 2
            if m == -j - 0.5:   # Extreme state. Edge left state
                spin_up = 0.0
                spin_down = 1.0
                zeeman_energy_khz = zeeman_edge_minus

                state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])

            else:               # Other non extreme states
                spin_up = sqrt((x - y) / (2 * x))
                spin_down = -sqrt((x + y) / (2 * x))
                zeeman_energy_khz = cij / 4 - self.cb_khz * gj * m + x

                state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])

        xi = True  # calculate xi = + states
        for m in self.m_csi_plus(j):
            x = 1 / 2 * sqrt(cij**2 * ((j + 1 / 2) ** 2 - m**2) + (cij * m - self.cb_khz * (gj - gI)) ** 2)
            y = -self.cb_khz / 2 * (gj - gI) + m * cij / 2

            if m == j + 0.5:        # Extreme state. Right edge state
                spin_up = 1.0
                spin_down = 0.0
                zeeman_energy_khz = zeeman_edge_plus

                state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])

            else:                   # Non-extreme states
                spin_up = sqrt((x + y) / (2 * x))
                spin_down = sqrt((x - y) / (2 * x))
                zeeman_energy_khz = cij / 4 - self.cb_khz * gj * m - x

                state_list.append([j, float(m), xi, spin_up, spin_down, zeeman_energy_khz, rotation_energy_ghz])

    """state dataframe"""
    self.state_df = pd.DataFrame(state_list, columns=self.state_df_columns)
