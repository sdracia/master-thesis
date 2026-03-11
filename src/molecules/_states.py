"""
State construction utilities for rotational-hyperfine manifolds.

This module defines helper functions to generate quantum numbers
for the two Xi manifolds and to initialize the full set of molecular states,
including Zeeman energies and spin composition coefficients.
"""

import numpy as np
import pandas as pd

from math import sqrt
from scipy.constants import physical_constants


gI = physical_constants["proton g factor"][0]


def m_csi_minus(cls, j: int) -> np.ndarray:
    """
    Return the quantum numbers m for a given J in the Xi = - manifold.

    Parameters
    ----------
    j : int
        Rotational quantum number J.

    Returns
    -------
    numpy.ndarray
        Array of m values ranging from -(J + 1/2) to J - 1/2 in steps of 1.
    """
    return np.arange(-j - 0.5, j, 1)


def m_csi_plus(cls, j: int) -> np.ndarray:
    """
    Return the quantum numbers m for a given J in the Xi = + manifold.

    Parameters
    ----------
    j : int
        Rotational quantum number J.

    Returns
    -------
    numpy.ndarray
        Array of m values ranging from -(J - 1/2) to J + 1/2 in steps of 1.
    """
    return np.arange(-j + 0.5, j + 1, 1)


def init_states(self) -> None:
    """
    Initialize all Zeeman-hyperfine states up to J_max.

    For each J = 0,...,J_max:
        - The Xi = - manifold is constructed first,
        - Followed by the Xi = + manifold.
        - Within each manifold, states are ordered by increasing m.

    For each state, the following quantities are computed:
        - Spin composition coefficients (spin_up, spin_down)
        - Zeeman energy (kHz)
        - Rotational energy (GHz)

    The resulting states are stored in ``self.state_df`` as a pandas DataFrame
    with columns defined by ``self.state_df_columns``.
    """

    state_list = []

    for i, j in enumerate(range(self.j_max + 1)):

        gj = self.gj_list[i] if self.gj_list != [] else self.gj
        cij = self.cij_list[i] if self.cij_list != [] else self.cij_khz

        rotation_energy_ghz = self.br_ghz * j * (j + 1)

        # J-dependent edge-state energies
        zeeman_edge_minus = (gj * j + gI / 2) * self.cb_khz - cij * j / 2
        zeeman_edge_plus = -(gj * j + gI / 2) * self.cb_khz - cij * j / 2

        # ------------------------
        # Xi = - manifold
        # ------------------------
        xi = False

        for m in self.m_csi_minus(j):

            x = 0.5 * sqrt(
                cij**2 * ((j + 0.5) ** 2 - m**2)
                + (cij * m - self.cb_khz * (gj - gI)) ** 2
            )
            y = -self.cb_khz * (gj - gI) / 2 + m * cij / 2

            if m == -j - 0.5:
                # Extreme left edge state: pure spin-down
                spin_up = 0.0
                spin_down = 1.0
                zeeman_energy_khz = zeeman_edge_minus
            else:
                spin_up = sqrt((x - y) / (2 * x))
                spin_down = -sqrt((x + y) / (2 * x))
                zeeman_energy_khz = cij / 4 - self.cb_khz * gj * m + x

            state_list.append(
                [
                    j,
                    float(m),
                    xi,
                    spin_up,
                    spin_down,
                    zeeman_energy_khz,
                    rotation_energy_ghz,
                ]
            )

        # ------------------------
        # Xi = + manifold
        # ------------------------
        xi = True

        for m in self.m_csi_plus(j):

            x = 0.5 * sqrt(
                cij**2 * ((j + 0.5) ** 2 - m**2)
                + (cij * m - self.cb_khz * (gj - gI)) ** 2
            )
            y = -self.cb_khz * (gj - gI) / 2 + m * cij / 2

            if m == j + 0.5:
                # Extreme right edge state: pure spin-up
                spin_up = 1.0
                spin_down = 0.0
                zeeman_energy_khz = zeeman_edge_plus
            else:
                spin_up = sqrt((x + y) / (2 * x))
                spin_down = sqrt((x - y) / (2 * x))
                zeeman_energy_khz = cij / 4 - self.cb_khz * gj * m - x

            state_list.append(
                [
                    j,
                    float(m),
                    xi,
                    spin_up,
                    spin_down,
                    zeeman_energy_khz,
                    rotation_energy_ghz,
                ]
            )

    self.state_df = pd.DataFrame(
        state_list,
        columns=self.state_df_columns,
    )