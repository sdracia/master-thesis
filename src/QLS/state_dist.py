# File: state_dist.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Utilities to calculate and store state distributions of molecules.

Includes:
- Thermal distributions for a given temperature.
- State distribution storage class with optional thermal weighting.
"""

from molecules.molecule import Molecule
from typing import Optional, NamedTuple
import numpy as np
from scipy.constants import h, k


class Polarization(NamedTuple):
    """
    Represents polarization of a light field.

    Attributes
    ----------
    pi : float
        π-polarized light.
    sp : float
        sigma+ polarized light.
    sm : float
        sigma- polarized light.
    """
    pi: float
    sp: float
    sm: float


def get_thermal_distribution(molecule: Molecule, temperature: float) -> np.ndarray:
    """
    Calculate the thermal population distribution over molecular states.

    Each state's population is weighted according to the Boltzmann factor
    based on its rotational energy.

    Parameters
    ----------
    molecule : Molecule
        The molecule whose states are used for the calculation.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Normalized thermal distribution over all molecular states.
    """
    rotational_energy_ghz = molecule.state_df["rotation_energy_ghz"].to_numpy()
    # Convert GHz to Hz for energy, then compute Boltzmann factor
    state_distribution = np.exp(-h * rotational_energy_ghz * 1e9 / (k * temperature))
    # Normalize
    state_distribution /= np.sum(state_distribution)
    return state_distribution


class States:
    """
    Stores and manages the state distribution of a molecule.

    Can initialize either a uniform distribution or a thermal distribution
    at a given temperature.
    """

    def __init__(self, molecule: Molecule, temperature: Optional[float] = None):
        """
        Initialize the States object.

        Parameters
        ----------
        molecule : Molecule
            The molecule for which to store the state distribution.
        temperature : float, optional
            Temperature in Kelvin. If provided, initializes the thermal distribution.
            If None, initializes a uniform distribution over all states.
        """
        self.molecule = molecule
        self.num_states = len(molecule.state_df)
        self.j = molecule.state_df["j"].to_numpy(dtype=int)

        if temperature is not None:
            self.dist = get_thermal_distribution(molecule, temperature)
        else:
            self.dist = np.ones(len(molecule.state_df)) / len(molecule.state_df)

        self.molecule.state_df["state_dist"] = self.dist

    def j_distribution(self) -> np.ndarray:
        """
        Compute the population distribution per rotational quantum number J.

        Returns
        -------
        np.ndarray
            Array of populations summed over all states with the same J.
        """
        return np.bincount(self.j, weights=self.dist)