import pandas as pd


def init_transition_dataframe(self):
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
        

        for index, state1 in enumerate(states_array):
            "1 state"
            index1 = states_index[index]
            m1, xi1, zeeman_energy_khz1 = state1[1], state1[2], state1[5]

            if index == 0:  
                """no transition from the Xi.minus left edge state"""
                continue

            if index == 1 or index == m_len:
                """the 2 states right next to the Xi.minus left edge state. They only have 1 transition with the left edge state"""
                index2 = states_index[0]
                state2 = states_array[0]
                m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]

                energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
                coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, 0, -1)
                transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])
                continue
            
            """For all the other states I have two transitions: one on the same value of csi, one on the opposite value of csi"""
            """Same value of csi"""
            index2 = states_index[index - 1]
            state2 = states_array[index - 1]
            m2, xi2, zeeman_energy_khz2 = state2[1], state2[2], state2[5]
            energy_diff = zeeman_energy_khz2 - zeeman_energy_khz1
            coupling = self.coupling_coefficient * self.get_raman_coupling(index1, index2, 0, -1)
            transition_list.append([j, m1, xi1, m2, xi2, index1, index2, energy_diff, coupling])

            """Opposite value of csi (depending on which csi has the initial state)"""
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

    """Transition Dataframe is built following this specific order of calculations"""
    self.transition_df = pd.DataFrame(transition_list, columns=self.transition_df_columns)
