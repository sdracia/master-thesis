import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_array as sparray
from scipy.sparse import diags, csr_array
from typing import Dict


from exp_imperfections import apply_noise
from ._utils import checks_likelihoods




def likelihoods_simulator(
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
) -> sparray:
    
    original_freq = frequency   


    if laser_miscalibration is None:
        laser_miscalibration = {}

    if "frequency" in laser_miscalibration:
        frequency = apply_noise(frequency, laser_miscalibration["frequency"]["type"], laser_miscalibration["frequency"]["level"], seed_miscalibration)
    if "rabi_rate" in laser_miscalibration:
        rabi_rate_mhz = apply_noise(rabi_rate_mhz, laser_miscalibration["rabi_rate"]["type"], laser_miscalibration["rabi_rate"]["level"], seed_miscalibration)

        
    if noise_params is None:
        noise_params = {}

    if "frequency" in noise_params:
        frequency = apply_noise(frequency, noise_params["frequency"]["type"], noise_params["frequency"]["level"], seed)
    if "rabi_rate" in noise_params:
        rabi_rate_mhz = apply_noise(rabi_rate_mhz, noise_params["rabi_rate"]["type"], noise_params["rabi_rate"]["level"], seed)
    

    self.misfrequency = frequency-original_freq       

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

    return exc_matrix_checked


def new_state_index(self, col_sparse, rng=np.random.default_rng()):
    """
    Samples a row index from the sparse column `col_sparse`, treating it as a probability distribution.

    Args:
        col_sparse: a sparse column vector (CSC or CSR format, shape (N, 1))
        rng: random number generator (default: np.random.default_rng())

    Returns:
        A row index sampled according to the probability distribution defined by the column values.
    """
    # Extract non-zero row indices and their corresponding values
    row_indices = col_sparse.nonzero()[0]
    values = col_sparse.data

    # print(f"Row indices: {row_indices}. Values: {values}")

    # Normalize the values to form a probability distribution
    prob_dist = values / values.sum()

    # Sample a row index using the probability distribution
    chosen_idx = rng.choice(row_indices, p=prob_dist)
    
    return chosen_idx