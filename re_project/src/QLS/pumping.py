import numpy as np
from molecules.molecule import Molecule
from scipy.constants import h, k
from scipy.sparse import csr_array, sparray
from typing import Tuple, Optional, NamedTuple
from typing import List, Dict
import exp_imperfections as imp


def excitation_matrix(
    molecule: Molecule,
    frequency: float,
    duration_us: float,
    rabi_rate_mhz: float,
    dephased: bool = False,
    coherence_time_us: float = 1000.0,
    is_minus: bool = True,
    noise_params: Dict[str, Dict[str, float]] = None,
    seed: int = None,
    laser_miscalibration: Dict[str, Dict[str, float]] = None,   
    seed_miscalibration: int = None
) -> sparray:
    """Returns the excitation probabilities for given frequency and other parameters

    Args:
        molecule (Molecule): The molecule to calculate the excitation probabilities for
        frequency (float): The frequency of the excitation pulse in MHz
        duration_us (float): The duration of the excitation pulse in microseconds
        rabi_rate_mhz (float): The Rabi rate in MHz
        dephased (bool): If True, the excitation is dephased
        coherence_time_us (float): The coherence time in us for rabi flopping
        is_minus (bool): If True, the excitation is for dm = -1
    Returns:
        np.ndarray: The excitation probabilities for each state
    """
        
    if laser_miscalibration is None:
        laser_miscalibration = {}

    if "frequency" in laser_miscalibration:
        frequency = imp.apply_noise(frequency, laser_miscalibration["frequency"]["type"], laser_miscalibration["frequency"]["level"], seed_miscalibration)
    if "rabi_rate" in laser_miscalibration:
        rabi_rate_mhz = imp.apply_noise(rabi_rate_mhz, laser_miscalibration["rabi_rate"]["type"], laser_miscalibration["rabi_rate"]["level"], seed_miscalibration)


    if noise_params is None:
        noise_params = {}

    # Applicare rumore separato per ogni parametro se specificato
    if "frequency" in noise_params:
        frequency = imp.apply_noise(frequency, noise_params["frequency"]["type"], noise_params["frequency"]["level"], seed)
    if "rabi_rate" in noise_params:
        rabi_rate_mhz = imp.apply_noise(rabi_rate_mhz, noise_params["rabi_rate"]["type"], noise_params["rabi_rate"]["level"], seed)


    num_states = len(molecule.state_df)


    if is_minus:
        detunings = 2 * np.pi * (frequency - molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)
    else:
        detunings = 2 * np.pi * (frequency + molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)

    # detunings = 2 * np.pi * (frequency - molecule.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)
    omegas = rabi_rate_mhz * molecule.transition_df["coupling"].to_numpy(dtype=float)

    if dephased:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * ((1 - np.cos(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us) * np.exp(-duration_us / coherence_time_us)) / 2)
    else:
        transition_exc_probs = omegas**2 / (omegas**2 + detunings**2) * np.sin(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us / 2) ** 2
        
    if is_minus:
        # state1 --> state2
        rows = molecule.transition_df["index2"].to_numpy(dtype=int)
        cols = molecule.transition_df["index1"].to_numpy(dtype=int)
        exc_matrix = csr_array((transition_exc_probs, (rows, cols)), shape=(num_states, num_states)) + csr_array((-transition_exc_probs, (cols, cols)), shape=(num_states, num_states))
    else:
        # state2 --> state1
        rows = molecule.transition_df["index1"].to_numpy(dtype=int)
        cols = molecule.transition_df["index2"].to_numpy(dtype=int)
        exc_matrix = csr_array((transition_exc_probs, (rows, cols)), shape=(num_states, num_states)) + csr_array((-transition_exc_probs, (cols, cols)), shape=(num_states, num_states))

    return exc_matrix



def apply_pumping(
    molecule: Molecule,
    pump_frequency_mhz: float,
    num_pumps: int,
    pump_duration_us: float,
    pump_rabi_rate_mhz: float,
    pump_dephased: bool = False,
    coherence_time_us: float = 1000.0,
    is_minus: bool = True,
    noise_params: Dict[str, Dict[str, float]] = None,
    seed: int = None,
    laser_miscalibration: Dict[str, Dict[str, float]] = None,   
    seed_miscalibration: int = None
) -> None:

    for _ in range(num_pumps):


        exc_matrix = excitation_matrix(molecule, pump_frequency_mhz, pump_duration_us, pump_rabi_rate_mhz, pump_dephased, coherence_time_us, is_minus, noise_params, seed, laser_miscalibration, seed_miscalibration).dot(molecule.state_df["state_dist"])
        molecule.state_df["state_dist"] += exc_matrix


        mask = molecule.state_df["state_dist"] < 0
        if np.abs(sum(exc_matrix)) >= 1e-10:
            raise ValueError("Error: sum of exc_matrix is not 0")

        if (molecule.state_df["state_dist"].shape) != (exc_matrix.shape):
            raise ValueError(f"Error: Shape mismatch. state_dist has shape {molecule.state_df['state_dist'].shape}, but exc_matrix has shape {exc_matrix.shape}")

        if np.sum(mask) > 0 : 
            raise ValueError("Error: state_dist contains negative values")

