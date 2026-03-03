from molecules.molecule import Molecule
from typing import Optional, NamedTuple
import numpy as np
from scipy.constants import h, k

class Polarization(NamedTuple):
    pi: float
    sp: float
    sm: float


def get_thermal_distribution(molecule: Molecule, temperature: float) -> np.ndarray:
    """Returns the thermal distribution for a given temperature

    Args:
        molecule (Molecule): The molecule to calculate the thermal distribution for
        temperature (float): The temperature in Kelvin
    Returns:
        np.ndarray: The thermal distribution for each state
    """
    rotational_energy_ghz = molecule.state_df["rotation_energy_ghz"].to_numpy()


    state_distribution = np.exp(-h * rotational_energy_ghz * 1e9 / (k * temperature))


    state_distribution /= np.sum(state_distribution)

    return state_distribution




class States:
    """Class to store the state distribution of the given molecule"""

    def __init__(self, molecule: Molecule, temperature: Optional[float] = None):
        """Initializes the States object

        Args:
            molecule (Molecule): The molecule to store the state distribution for
            temperature (float): The temperature in Kelvin, if None, the state distribution is uniform
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
        """Returns the distribution of the rotational states"""
        return np.bincount(self.j, weights=self.dist)