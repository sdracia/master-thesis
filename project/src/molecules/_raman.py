from wigners import clebsch_gordan
import numpy as np



def get_raman_coupling(self, index1, index2, qa, qb):
    """
    Calculates the coupling between two Zeeman levels, |m1,xi1> & |m2,xi2>
    with two Raman beam with polarization qa and qb.
    Detuning and the electronic transition term is not considered.
    """


    def j_coupling(j, j_exc, mj1, mj2, qa, qb) -> float:
        """
        Function which computes the coupling coefficients based on the selection rules on j and m.
        Input parameters: 
        - j: the value of j of the initial and final state of the transition (both states belong to the same manifold)
        - j_exc: excited value of j due to the Raman beam. It can be j+1 or j-1
        - mj1: projection of rotational angular momentum for initial level
        - mj2: projection of rotational angular momentum for final level
        - qa: polarization of the first beam
        - qb: polarization of the second beam
        """
        if j < 0 or j_exc < 0:
            return 0
        if j < abs(mj1) or j < abs(mj2) or j_exc < abs(mj1 + qa):
            return 0
        return (
            np.sqrt((2 * j_exc + 1) / (2 * j + 1))      # """First transition: initial --> excited """
            * clebsch_gordan(1, 0, j_exc, 0, j, 0)
            * clebsch_gordan(1, -qa, j_exc, mj1 + qa, j, mj1)   

            * np.sqrt((2 * j + 1) / (2 * j_exc + 1))    # """Second transition: excited --> final"""  
            * clebsch_gordan(1, 0, j, 0, j_exc, 0)
            * clebsch_gordan(1, -qb, j, mj2, j_exc, mj1 + qa)
        )


    state1 = self.state_df.loc[index1]
    state2 = self.state_df.loc[index2]
    j = state1.j                  
    m1 = state1.m               
    m2 = state2.m
    m1_up = int(m1 - 0.5)       # è mj = m - 1/2, quindi è l'm per lo stato spin_up, ossia quello con mI = 1/2
    m1_down = int(m1 + 0.5)     # è mj = m + 1/2, quindi è l'm per lo stato spin_down, ossia quello con mI = -1/2
    m2_up = int(m2 - 0.5)       # stessa cosa per il secondo stato 
    m2_down = int(m2 + 0.5)

    # counter-rotating terms S-
    """
    Counter-rotating terms S- : I consider all combinations of spin_up and spin_down between first and second states.
    Then it is weighted according to the formula in Chou et al. 
    Both for J+1 and J-1
    Combinations of spin_up and spin_down are neglected, since mI is conserved in the transition
    """
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

    """
    co-rotating terms S+ : I consider all combinations of spin_up and spin_down between first and second states.
    The path is inversed: initial polarization is qa, second polarization is qb
    Then it is weighted according to the formula in Chou et al. 
    Both for J+1 and J-1
    Combinations of spin_up and spin_down are neglected, since I is conserved in the transition
    """
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


    """The result is a weighted mean of the co-rotating and counter-rotating terms"""
    return (coupling_minus + coupling_plus) / (1.0 / (self.omega_thz - self.omega_0_thz) + 1.0 / (self.omega_0_thz + self.omega_thz))
    # return (coupling_minus + coupling_plus) 
