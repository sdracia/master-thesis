"""
Transition construction for Raman-coupled Zeeman states.

This module defines the routine that builds the transition DataFrame
for all states within each fixed-J manifold. Transitions are constructed
according to the ordering convention used in `init_states`.
"""

import pandas as pd


def init_transition_dataframe(self) -> None:
    """
    Initialize the transition DataFrame for all J manifolds.

    For each fixed J:
        - States are assumed to be ordered as:
            1. Xi = - manifold (in increasing m),
            2. Xi = + manifold (in increasing m).
        - Allowed Raman transitions are constructed between:
            (i) Adjacent states within the same Xi manifold,
            (ii) States belonging to opposite Xi manifolds,
                according to the indexing structure.

    For each transition, the following quantities are stored:
        - j : rotational quantum number
        - m1, xi1 : initial state quantum numbers
        - m2, xi2 : final state quantum numbers
        - index1, index2 : indices in `state_df`
        - energy_diff : Zeeman energy difference (kHz)
        - coupling : Raman coupling strength

    The resulting transitions are stored in `self.transition_df`.
    """

    transition_list = []

    for j in range(self.j_max + 1):

        # Select states belonging to fixed J
        states_in_j = self.state_df.loc[self.state_df["j"] == j]
        states_index = states_in_j.index.to_numpy()
        states_array = states_in_j.to_numpy()

        # Number of m values in each Xi manifold
        m_len = 2 * j + 1

        for index, state1 in enumerate(states_array):

            index1 = states_index[index]
            m1 = state1[1]
            xi1 = state1[2]
            zeeman_energy_khz1 = state1[5]

            # No transition from the leftmost Xi = - edge state
            if index == 0:
                continue

            # States adjacent to the left edge only connect to that edge
            if index == 1 or index == m_len:

                index2 = states_index[0]
                state2 = states_array[0]

                m2 = state2[1]
                xi2 = state2[2]
                zeeman_energy_khz2 = state2[5]

                energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                coupling = (
                    self.coupling_coefficient
                    * self.get_raman_coupling(index1, index2, 0, -1)
                )

                transition_list.append(
                    [j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling]
                )

                continue

            # ------------------------------------------
            # Same Xi manifold (adjacent in ordering)
            # ------------------------------------------
            index2 = states_index[index - 1]
            state2 = states_array[index - 1]

            m2 = state2[1]
            xi2 = state2[2]
            zeeman_energy_khz2 = state2[5]

            energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
            coupling = (
                self.coupling_coefficient
                * self.get_raman_coupling(index1, index2, 0, -1)
            )

            transition_list.append(
                [j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling]
            )

            # ------------------------------------------
            # Opposite Xi manifold
            # ------------------------------------------
            if xi1:  # Xi = +
                index2 = states_index[index - m_len]
            else:    # Xi = -
                index2 = states_index[index + m_len - 2]

            state2 = states_array[
                index - m_len if xi1 else index + m_len - 2
            ]

            m2 = state2[1]
            xi2 = state2[2]
            zeeman_energy_khz2 = state2[5]

            energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
            coupling = (
                self.coupling_coefficient
                * self.get_raman_coupling(index1, index2, 0, -1)
            )

            transition_list.append(
                [j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling]
            )

    self.transition_df = pd.DataFrame(
        transition_list,
        columns=self.transition_df_columns,
    )