# File: __init__.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Public API for the molecular spectroscopy package.

This module re-exports the main molecular classes and core utility
functions for state initialization, transition construction, Raman
coupling evaluation, and Zeeman level visualization.

The ``__all__`` variable defines the public interface of the package.
"""

from .molecule import CaH, CaOH, CaOH_dm2, CaH_dm2, Molecule

from ._states import m_csi_minus, m_csi_plus, init_states
from ._transitions import init_transition_dataframe
from ._raman import get_raman_coupling
from ._plotting import plot_zeeman_levels
from ._dm2 import (
    init_transition_dm2_dataframe,
    plot_zeeman_levels_dm2,
)

__all__ = [
    "CaH",
    "CaOH",
    "CaOH_dm2",
    "CaH_dm2",
    "Molecule",
    "m_csi_minus",
    "m_csi_plus",
    "init_states",
    "init_transition_dataframe",
    "get_raman_coupling",
    "plot_zeeman_levels",
    "init_transition_dm2_dataframe",
    "plot_zeeman_levels_dm2",
]