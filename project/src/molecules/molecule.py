"""
Molecular structure definitions.

This module defines object-oriented representations of molecules,
including Zeeman structure, rotational constants, Raman coupling parameters,
and transition data management. It provides base and dm2 variants, which differ
in the way transition data are initialized and handled.
"""

import pandas as pd
from typing import TypeAlias
from scipy.constants import h, physical_constants

# Physical constants
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
from ._io import (
    read_molecule_data,
    create_molecule_data,
    read_molecule_data_dm2,
    create_molecule_data_dm2,
    save_data,
)


class CaH:
    """
    Base class describing the CaH molecule.

    This class stores molecular constants, initializes Zeeman states and
    transition data structures, and exposes methods for Raman coupling,
    plotting, and I/O operations.

    Parameters
    ----------
    b_field_gauss : float
        External magnetic field in Gauss.
    j_max : int
        Maximum rotational quantum number J included in the model.
    gj_list : list[float], optional
        Optional list of g_J factors for each J = 0,...,j_max.
        If provided, must have length j_max + 1.
    cij_list : list[float], optional
        Optional list of spin-rotation coupling constants (kHz)
        for each J = 0,...,j_max. If provided, must have length j_max + 1.
        
    Physical constants (class attributes)
    -------------------------------------
    name : str
        Molecule name.
    gj : float
        Rotational g-factor.
    cij_khz : float
        Spin-rotation coupling constant (kHz).
    br_ghz : float
        Rotational constant B (GHz).
    omega_0_thz : float
        Electronic transition frequency (THz).
    omega_thz : float
        Raman laser frequency (THz).
    coupling_coefficient : float
        Dimensionless scaling factor for Raman coupling strength.
    """

    name: str = "CaH"
    gj: float = -1.36
    cij_khz: float = 8.52
    br_ghz: float = 142.5017779
    omega_0_thz: float = 750.0
    omega_thz: float = 285.5
    coupling_coefficient: float = 1.0

    def __init__(
        self,
        b_field_gauss: float,
        j_max: int,
        gj_list: list[float] = None,
        cij_list: list[float] = None,
    ) -> None:

        self.b_field_gauss = b_field_gauss
        self.j_max = j_max

        # If provided, these override the single-value constants per J
        self.gj_list = gj_list if gj_list is not None else []
        self.cij_list = cij_list if cij_list is not None else []

        if self.gj_list and self.cij_list:
            if len(self.gj_list) != (j_max + 1) or len(self.cij_list) != (j_max + 1):
                raise ValueError(
                    "Wrong input dimensions for j_max, cij_list, gj_list"
                )

        # Zeeman coefficient: μ_N B / h converted to kHz
        self.cb_khz = mu_N * b_field_gauss * 1e-4 / h / 1e3

        # DataFrame storing Zeeman states
        self.state_df: pd.DataFrame = pd.DataFrame()
        self.state_df_columns = [
            "j",
            "m",
            "xi",
            "spin_up",
            "spin_down",
            "zeeman_energy_khz",
            "rotation_energy_ghz",
        ]

        # DataFrame storing allowed transitions
        self.transition_df: pd.DataFrame = pd.DataFrame()
        self.transition_df_columns = [
            "j",
            "m1",
            "xi1",
            "m2",
            "xi2",
            "index1",
            "index2",
            "energy_diff",
            "coupling",
        ]

    # Bind external functionality as class methods or attributes
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
    """
    CaOH molecule definition.

    Inherits from CaH and overrides molecular constants.
    """

    name: str = "CaOH"
    gj: float = -0.036
    cij_khz: float = 1.49
    br_ghz: float = 10.9921442
    omega_0_thz: float = 1100.0
    omega_thz: float = 280.1869
    coupling_coefficient: float = 1.0

    def __init__(
        self,
        b_field_gauss: float,
        j_max: int,
        gj_list: list[float] = None,
        cij_list: list[float] = None,
    ) -> None:
        super().__init__(b_field_gauss, j_max, gj_list, cij_list)


class CaOH_dm2(CaH):
    """
    CaOH molecule with dm2 transition formalism.

    This variant uses alternative transition initialization and plotting
    routines based on the dm2 implementation.
    """

    name: str = "CaOH"
    gj: float = -0.036
    cij_khz: float = 1.49
    br_ghz: float = 10.9921442
    omega_0_thz: float = 1100.0
    omega_thz: float = 280.1869
    coupling_coefficient: float = 1.0

    def __init__(
        self,
        b_field_gauss: float,
        j_max: int,
        gj_list: list[float] = None,
        cij_list: list[float] = None,
    ) -> None:
        super().__init__(b_field_gauss, j_max, gj_list, cij_list)

    read_molecule_data_dm2 = classmethod(read_molecule_data_dm2)
    create_molecule_data_dm2 = classmethod(create_molecule_data_dm2)

    init_transition_dm2_dataframe = init_transition_dm2_dataframe
    plot_zeeman_levels_dm2 = plot_zeeman_levels_dm2


class CaH_dm2(CaH):
    """
    CaH molecule with dm2 transition formalism.
    """

    name: str = "CaH"
    gj: float = -1.36
    cij_khz: float = 8.52
    br_ghz: float = 142.5017779
    omega_0_thz: float = 750.0
    omega_thz: float = 285.5
    coupling_coefficient: float = 1.0

    def __init__(
        self,
        b_field_gauss: float,
        j_max: int,
        gj_list: list[float] = None,
        cij_list: list[float] = None,
    ) -> None:
        super().__init__(b_field_gauss, j_max, gj_list, cij_list)

    read_molecule_data_dm2 = classmethod(read_molecule_data_dm2)
    create_molecule_data_dm2 = classmethod(create_molecule_data_dm2)

    init_transition_dm2_dataframe = init_transition_dm2_dataframe
    plot_zeeman_levels_dm2 = plot_zeeman_levels_dm2


Molecule: TypeAlias = CaH | CaOH | CaOH_dm2 | CaH_dm2