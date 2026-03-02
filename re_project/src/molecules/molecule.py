import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from typing import TypeAlias
from pathlib import Path
from wigners import clebsch_gordan
from scipy.constants import h, physical_constants

# physical constants
mu_N = physical_constants["nuclear magneton"][0]
gI = physical_constants["proton g factor"][0]



from ._states import m_csi_minus, m_csi_plus, init_states
from ._transitions import init_transition_dataframe
from ._raman import get_raman_coupling
from ._plotting import plot_zeeman_levels
from ._dm2 import (
    init_transition_dm2_dataframe,
    plot_zeeman_levels_dm2,
)
from ._io import read_molecule_data, create_molecule_data, read_molecule_data_dm2, create_molecule_data_dm2, save_data





class CaH:
    name: str = "CaH"
    """name of the molecule"""
    
    gj: float = -1.36
    """g factor for J"""
    
    cij_khz: float = 8.52   # kHz
    """coupling strength between proton spin and molecule rotation, in kHz"""

    br_ghz: float = 142.5017779
    """rotational constant, in GHz"""
    
    omega_0_thz: float = 750.0      # THz
    """frequency of the electronic transition between ground state and 1st excited state, in THz"""
    omega_thz: float = 285.5        # THz
    """frequency of the Raman beam, in THz"""

    coupling_coefficient: float = 1.
    """coupling coefficient"""

    # CONSTRUCTOR
    def __init__(self, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None) -> None:
        self.b_field_gauss = b_field_gauss
        """magnetic field in Gauss"""

        self.j_max = j_max
        """maximum j value to consider"""

        self.gj_list = gj_list if gj_list is not None else []
        self.cij_list = cij_list if cij_list is not None else []
        """If the lists exist, they are taken as gj and cij for different j's, otherwise the single value is considered"""

        if self.gj_list and self.cij_list:
            if len(self.gj_list) != (j_max +1) or len(self.cij_list) != (j_max +1):
                raise ValueError("Wrong input dimensions for j_max, cij_list, gj_list")

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

    read_molecule_data = classmethod(read_molecule_data)
    create_molecule_data = classmethod(create_molecule_data)
    m_csi_minus = m_csi_minus
    m_csi_plus = m_csi_plus
    save_data = save_data
    init_states = init_states
    init_transition_dataframe = init_transition_dataframe
    get_raman_coupling = get_raman_coupling
    plot_zeeman_levels = plot_zeeman_levels




class CaOH(CaH):
    name: str = "CaOH"
    """name of the molecule"""
    gj: float = -0.036
    """g factor for J"""
    cij_khz: float = 1.49
    """coupling strength between proton spin and molecule rotation, in kHz"""
    # br_ghz: float = 10.96    
    br_ghz: float = 10.9921442
    """rotational constant, in GHz"""
    # omega_0_thz: float = 118.49
    omega_0_thz: float = 1100.0
    """frequency of the electronic transition between ground state and 1st excited state, in THz"""
    omega_thz: float = 280.1869
    """frequency of the Raman beam, in THz"""
    coupling_coefficient: float = 1.
    """coupling coefficient, for now to be able to compare with the NIST value"""

    def __init__(self, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None) -> None:
        super().__init__(b_field_gauss, j_max, gj_list, cij_list)




class CaOH_dm2(CaH):
    name: str = "CaOH"
    """name of the molecule"""
    gj: float = -0.036
    """g factor for J"""
    cij_khz: float = 1.49
    """coupling strength between proton spin and molecule rotation, in kHz"""
    # br_ghz: float = 10.96    
    br_ghz: float = 10.9921442
    """rotational constant, in GHz"""
    # omega_0_thz: float = 118.49
    omega_0_thz: float = 1100.0
    """frequency of the electronic transition between ground state and 1st excited state, in THz"""
    omega_thz: float = 280.1869
    """frequency of the Raman beam, in THz"""
    coupling_coefficient: float = 1.
    """coupling coefficient, for now to be able to compare with the NIST value"""

    def __init__(self, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None) -> None:
        super().__init__(b_field_gauss, j_max, gj_list, cij_list)


    read_molecule_data_dm2 = classmethod(read_molecule_data_dm2)
    create_molecule_data_dm2 = classmethod(create_molecule_data_dm2)

    init_transition_dm2_dataframe = init_transition_dm2_dataframe
    plot_zeeman_levels_dm2 = plot_zeeman_levels_dm2



class CaH_dm2(CaH):
    name: str = "CaH"
    """name of the molecule"""
    
    gj: float = -1.36
    """g factor for J"""
    
    cij_khz: float = 8.52   # kHz
    """coupling strength between proton spin and molecule rotation, in kHz"""

    br_ghz: float = 142.5017779
    """rotational constant, in GHz"""
    
    omega_0_thz: float = 750.0      # THz
    """frequency of the electronic transition between ground state and 1st excited state, in THz"""
    omega_thz: float = 285.5        # THz
    """frequency of the Raman beam, in THz"""

    coupling_coefficient: float = 1.
    """coupling coefficient, for now to be able to compare with the NIST value"""

    def __init__(self, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None) -> None:
        super().__init__(b_field_gauss, j_max, gj_list, cij_list)


    read_molecule_data_dm2 = classmethod(read_molecule_data_dm2)
    create_molecule_data_dm2 = classmethod(create_molecule_data_dm2)

    init_transition_dm2_dataframe = init_transition_dm2_dataframe
    plot_zeeman_levels_dm2 = plot_zeeman_levels_dm2




Molecule: TypeAlias = CaH | CaOH | CaOH_dm2 | CaH_dm2
