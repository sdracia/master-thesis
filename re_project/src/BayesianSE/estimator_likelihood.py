import numpy as np
from scipy.sparse import diags, csr_array
from scipy.sparse import csr_matrix as sparray
from typing import Dict, Tuple

from exp_imperfections import apply_noise
from ._utils import checks_likelihoods


def likelihoods_estimator(
    self,
    frequency: float,
    duration_us: float,
    rabi_rate_mhz: float,
    dephased: bool = False,
    coherence_time_us: float = 1000.0,
    is_minus: bool = True,
    noise_params: Dict[str, Dict[str, float]] = None,
    seed: int = None,
    maximum_excitation: float = 0.9,
    laser_miscalibration: Dict[str, Dict[str, float]] = None,   
    seed_miscalibration: int = None
) -> Tuple[sparray, sparray]:

    # LASER MISCALIBRATION
    if laser_miscalibration is None:
        laser_miscalibration = {}

    if "frequency" in laser_miscalibration:
        frequency = apply_noise(frequency, laser_miscalibration["frequency"]["type"], laser_miscalibration["frequency"]["level"], seed_miscalibration)
    if "rabi_rate" in laser_miscalibration:
        rabi_rate_mhz = apply_noise(rabi_rate_mhz, laser_miscalibration["rabi_rate"]["type"], laser_miscalibration["rabi_rate"]["level"], seed_miscalibration)

    ## SHOT-TO-SHOT FLUCTUATIONS
    if noise_params is None:
        noise_params = {}

    if "frequency" in noise_params:
        frequency = apply_noise(frequency, noise_params["frequency"]["type"], noise_params["frequency"]["level"], seed)
    if "rabi_rate" in noise_params:
        rabi_rate_mhz = apply_noise(rabi_rate_mhz, noise_params["rabi_rate"]["type"], noise_params["rabi_rate"]["level"], seed)

    num_states = len(self.model.state_df)


    if is_minus:
        detunings = 2 * np.pi * (frequency - self.model.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)
    else:
        detunings = 2 * np.pi * (frequency + self.model.transition_df["energy_diff"].to_numpy(dtype=float) * 1e-3)

    omegas = rabi_rate_mhz * self.model.transition_df["coupling"].to_numpy(dtype=float)

    if dephased:
        transition_exc_probs = maximum_excitation * omegas**2 / (omegas**2 + detunings**2) * ((1 - np.cos(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us) * np.exp(-duration_us / coherence_time_us)) / 2)
    else:
        transition_exc_probs = maximum_excitation * omegas**2 / (omegas**2 + detunings**2) * np.sin(np.sqrt(omegas**2.0 + detunings**2.0) * duration_us / 2) ** 2
        
    if is_minus:
        # state1 --> state2
        rows = self.model.transition_df["index2"].to_numpy(dtype=int)
        cols = self.model.transition_df["index1"].to_numpy(dtype=int)
    else:
        # state2 --> state1
        rows = self.model.transition_df["index1"].to_numpy(dtype=int)
        cols = self.model.transition_df["index2"].to_numpy(dtype=int)
    
    exc_matrix = (
        diags([1.0] * num_states, offsets=0, format="csr") +
        csr_array((transition_exc_probs, (rows, cols)), shape=(num_states, num_states)) +
        csr_array((-transition_exc_probs, (cols, cols)), shape=(num_states, num_states))
    )

    exc_matrix_checked = checks_likelihoods(exc_matrix)

    # === Extract diagonal matrix and off-diagonal matrix ===
    diagonal_matrix = diags(exc_matrix_checked.diagonal(), format='csr')
    off_diagonal_matrix = exc_matrix_checked - diagonal_matrix

    likelihood0 = diagonal_matrix
    likelihood1 = off_diagonal_matrix

    return likelihood0, likelihood1
