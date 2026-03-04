"""
Raman coupling evaluation for Zeeman-hyperfine states.

This module computes the effective Raman coupling between two molecular
Zeeman states and, accounting for spin projections
(spin_up/spin_down) and both J -> J+1 and J -> J-1 virtual transitions.

The calculation is based on Clebsch-Gordan coefficients following
the prescription in Chou et al., neglecting detuning and electronic
transition prefactors beyond the energy denominators.
"""

from wigners import clebsch_gordan
import numpy as np


def get_raman_coupling(self, index1, index2, qa, qb) -> float:
    """
    Compute the Raman coupling between two Zeeman-hyperfine states.

    Parameters
    ----------
    index1 : int
        Index of the initial state in self.state_df.
    index2 : int
        Index of the final state in self.state_df.
    qa : int
        Polarization of the first Raman beam.
    qb : int
        Polarization of the second Raman beam.

    Returns
    -------
    float
        Effective Raman coupling between the two states.
        Includes contributions from co-rotating (S+) and counter-rotating (S-) paths.
    """

    def j_coupling(j, j_exc, mj1, mj2, qa, qb) -> float:
        """
        Compute the Clebsch-Gordan-based coupling for a single virtual transition.

        Parameters
        ----------
        j : int
            Rotational quantum number of initial/final states (within same manifold).
        j_exc : int
            Virtual rotational quantum number of the intermediate state (J ± 1).
        mj1 : int
            Projection of rotational angular momentum for initial spin state.
        mj2 : int
            Projection of rotational angular momentum for final spin state.
        qa : int
            Polarization of the first Raman beam.
        qb : int
            Polarization of the second Raman beam.

        Returns
        -------
        float
            Clebsch-Gordan-weighted coupling, or 0 if selection rules forbid transition.
        """
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

    # Extract state information
    state1 = self.state_df.loc[index1]
    state2 = self.state_df.loc[index2]

    j = state1.j
    m1 = state1.m
    m2 = state2.m

    # Projections for spin-up (mI=+1/2) and spin-down (mI=-1/2)
    m1_up = int(m1 - 0.5)
    m1_down = int(m1 + 0.5)
    m2_up = int(m2 - 0.5)
    m2_down = int(m2 + 0.5)

    # -------------------------
    # Counter-rotating terms S-
    # -------------------------
    coupling_minus = (
        1.0 / (self.omega_thz - self.omega_0_thz)
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

    # -------------------------
    # Co-rotating terms S+
    # -------------------------
    coupling_plus = (
        1.0 / (self.omega_0_thz + self.omega_thz)
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

    # Weighted mean of co-rotating and counter-rotating contributions
    return (coupling_minus + coupling_plus) / (
        1.0 / (self.omega_thz - self.omega_0_thz) + 1.0 / (self.omega_0_thz + self.omega_thz)
    )