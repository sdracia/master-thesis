"""
Utilities for Rapid Adiabatic Passage (RAP) Transition Analysis and Simulation.

This module provides support functions for:
- Computing transition selections from molecular dataframes.
- Preparing simulation arguments and coordinate mapping for J-manifolds.
- Constructing initial density matrices for combined internal-motional spaces.
- Visualizing pulse envelopes and local adiabaticity conditions.
- Generating Bloch sphere animations for quantum state evolution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from qutip import *
from typing import Tuple, List, Dict, Any, Union, Optional

from saving import save_figure_in_images


def compute_transitions(
    molecule: Any, 
    j_max: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes and filters specific molecular transitions across rotational manifolds.

    This function extracts four specific sets of transitions based on selection 
    rules and specific m-state indices (left/right positioning).

    Parameters
    ----------
    molecule : Molecule
        The molecule object containing the transition data in 'transition_df'.
    j_max : int
        The maximum value of j to consider for the extraction.

    Returns
    -------
    left_states_transitions : pd.DataFrame
        Transitions between lower manifold left states and upper states (dm = -1).
    left_to_right_transitions : pd.DataFrame
        Transitions between lower manifold left states and upper states (dm = +1).
    signature_transitions_left : pd.DataFrame
        Penultimate left state transitions within upper manifolds (dm = -1).
    signature_transitions_right : pd.DataFrame
        Penultimate left state transitions within upper manifolds (dm = +1).
    """
    left_states_transitions = []
    left_to_right_transitions = []
    signature_transitions_left = []
    signature_transitions_right = []

    for j_val in range(0, j_max + 1):
        # Filter transitions for the current manifold
        transitions_in_j = molecule.transition_df[molecule.transition_df["j"] == j_val]

        # Specific selection based on xi flags (parity/symmetry indicators)
        filtered_transitions = transitions_in_j[
            (transitions_in_j["xi1"] == True) & (transitions_in_j["xi2"] == False)
        ]

        # Selection of the first available transition for signature
        signature_transition_left = transitions_in_j.iloc[0]
        signature_transitions_left.append(signature_transition_left)
        
        if not filtered_transitions.empty:
            left_state_trans = filtered_transitions.iloc[0]
            left_states_transitions.append(left_state_trans)
            
        if len(transitions_in_j) > 2:
            # Specific indexing for right-hand side transitions
            left_to_right_trans = transitions_in_j.iloc[2]
            left_to_right_transitions.append(left_to_right_trans)

            signature_transition_right = filtered_transitions.iloc[1]
            signature_transitions_right.append(signature_transition_right)

    # Convert lists back to DataFrames
    left_states_transitions = pd.DataFrame(left_states_transitions)
    left_to_right_transitions = pd.DataFrame(left_to_right_transitions)
    signature_transitions_left = pd.DataFrame(signature_transitions_left)
    signature_transitions_right = pd.DataFrame(signature_transitions_right)

    return (
        left_states_transitions, 
        left_to_right_transitions, 
        signature_transitions_left, 
        signature_transitions_right
    )


def RAP_args(
    dataframe: pd.DataFrame, 
    is_minus: bool, 
    n_motional: int, 
    rabi_rate: float, 
    laser_detuning: float, 
    times: np.ndarray, 
    T: float, 
    D: float, 
    sigma: float, 
    trap_freq: Optional[float] = None, 
    lamb_dicke: Optional[float] = None, 
    off_resonant: bool = False
) -> Tuple[Dict[str, List[Dict[str, Any]]], pd.DataFrame]:
    """
    Prepares the argument dictionary for Master Equation (mesolve) RAP simulations.

    Also augments the input DataFrame with internal label indexing for 
    population tracking.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Transitions data including energy levels and couplings.
    is_minus : bool
        Determines the sign of the molecular transition frequency.
    n_motional : int
        Fock space dimension for motional states.
    rabi_rate : float
        Peak Rabi frequency.
    laser_detuning : float
        Central detuning of the laser.
    times : np.ndarray
        Simulation time array.
    T : float
        Pulse duration.
    D : float
        Sweep range for frequency chirp.
    sigma : float
        Gaussian envelope standard deviation.
    trap_freq : float, optional
        Ion trap frequency.
    lamb_dicke : float, optional
        Lamb-Dicke parameter.
    off_resonant : bool, optional
        If True, includes carrier interference terms. Default is False.

    Returns
    -------
    args : dict
        Nested dictionary with parameters for each J transition.
    rescaled_df : pd.DataFrame
        Dataframe with added 'label_1' (lower state) and 'label_2' (upper state).
    """
    H_terms_args = []
    rescaled_df = dataframe.reset_index(drop=True).copy()

    # Assign internal state labels based on row index for tracking
    rescaled_df['label_1'] = rescaled_df.index * 2        
    rescaled_df['label_2'] = rescaled_df.index * 2 + 1

    for _, row in rescaled_df.iterrows():
        j = int(row["j"])
        coupling = row["coupling"]

        # Invert energy difference if the transition direction is reversed
        if is_minus:
            w_mol = row["energy_diff"] * 1e-3
        else:
            w_mol = -row["energy_diff"] * 1e-3

        # Consolidate parameters into a dictionary for each manifold
        term_args = {
            'n_motional': int(n_motional), 
            'j': int(j),
            'rabi_rate': rabi_rate,
            'coupling': np.abs(coupling), 
            'w_mol': 2 * np.pi * w_mol, 
            'laser_detuning': laser_detuning,
            'trap_freq': trap_freq,
            'lamb_dicke': lamb_dicke,
            'off_resonant': off_resonant,
            'final_time': times[-1],
            'T': T,
            'D': D,
            'sigma': sigma
        }

        H_terms_args.append(term_args)
    
    args = {'terms': H_terms_args}

    return args, rescaled_df


def RAP_dm(
    term_args: Dict[str, Any], 
    n_motional: int, 
    from_simulation: bool = False, 
    sideband: bool = True, 
    init_pop_list: List[float] = [0.5, 0.5]
) -> Qobj:
    """
    Constructs the initial density matrix for a specific RAP transition.

    Parameters
    ----------
    term_args : dict
        Parameters for the specific transition.
    n_motional : int
        Motional Hilbert space size.
    from_simulation : bool, optional
        Flag for importing states from prior results (placeholder logic).
    sideband : bool, optional
        True for BSB transitions, False for Carrier.
    init_pop_list : list of float, optional
        Initial populations for the internal states [p0, p1].

    Returns
    -------
    rho : Qobj
        Combined density matrix (internal ⊗ motion).
    """
    if from_simulation:
        # Placeholder for complex state recovery from full molecular simulations
        j = term_args['j']
        population_init = term_args['population_init']
        population_fin = term_args['population_fin']
        # Note: Reduced density matrix logic to be implemented
        rho = None 

    else:
        # basis(2,0) is ground, basis(2,1) is excited
        if sideband:
            assert np.isclose(sum(init_pop_list), 1.0), "Populations must sum to 1"
            assert len(init_pop_list) == 2, "Only two-level systems are supported"

            states = [basis(2, i) for i in range(2)]
            rho_internal = sum([p * ket2dm(s) for p, s in zip(init_pop_list, states)])

            # Initial motional state is the vacuum state |0>
            rho_motional = ket2dm(basis(n_motional, 0))
        else:
            # For carrier transitions, start in pure ground state
            rho_internal = ket2dm(basis(2, 0))
            rho_motional = ket2dm(basis(n_motional, 0))
            
        rho = tensor(rho_internal, rho_motional)

    print("Density Matrix created")
    return rho


def chirp_envelope(
    args: Dict[str, Any], 
    times: np.ndarray, 
    T: float, 
    D: float, 
    sigma: float, 
    rabi_rate: float, 
    laser_detuning: float, 
    j_plot: Optional[Union[int, List[int]]] = None, 
    savetext: str = "rap_pulse"
) -> None:
    """
    Plots Rabi rate, frequency detuning, and local adiabaticity over time.

    Parameters
    ----------
    args : dict
        Nested simulation parameters.
    times : np.ndarray
        Time grid for evaluation.
    T : float
        Pulse duration.
    D : float
        Chirp range.
    sigma : float
        Gaussian envelope width.
    rabi_rate : float
        Peak Rabi frequency.
    laser_detuning : float
        Fixed detuning.
    j_plot : int or list, optional
        Selection of manifolds to visualize.
    savetext : str, optional
        File prefix for saved images.
    """
    available_j = [term["j"] for term in args["terms"]]
    J_max = max(available_j)

    if j_plot is None:
        j_to_plot = available_j
    elif isinstance(j_plot, int):
        j_to_plot = [j_plot]
    else:
        j_to_plot = list(j_plot)

    for j in j_to_plot:
        if j not in available_j:
            raise ValueError(f"Requested J={j} not available. Range: {min(available_j)}-{J_max}")

    for arg in args['terms']:
        j = arg['j']
        if j not in j_to_plot:
            continue

        w_mol = arg['w_mol']

        # Time-dependent Rabi rate (Gaussian)
        omega_t = rabi_rate * np.exp(-(times - T / 2)**2 / (2 * sigma**2))

        # Time-dependent Detuning (Linear Chirp)
        delta_t = D / T * (times - T / 2) + laser_detuning - w_mol

        # Local adiabaticity calculation as per standard RAP theory
        paper_adiabaticity = (
            np.sqrt(omega_t**2 * (times - T / 2)**2 / sigma**4 + D**2 / T**2) / 
            (omega_t**2 + delta_t**2)
        )

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 4))

        # --- Subplot Left: Rabi Rate and Detuning ---
        color1 = 'tab:blue'
        ax1.set_xlabel('Time (ms)', fontsize=20)
        ax1.set_ylabel('Rabi rate (kHz)', color=color1, fontsize=20)
        ax1.plot(times * 1e-3, omega_t * 1e3 / (2 * np.pi), color=color1, label='Rabi Rate')
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=20)
        ax1.tick_params(axis='x', labelsize=20)
        ax1.grid()

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Detuning (kHz)', color=color2, fontsize=20)
        ax2.plot(times * 1e-3, delta_t * 1e3 / (2 * np.pi), color=color2, label='Detuning')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=20)

        # --- Subplot Right: Adiabaticity ---
        ax3.set_title("Adiabaticity Conditions")
        ax3.plot(times * 1e-3, paper_adiabaticity, label="Local Adiabaticity", color="blue")
        ax3.set_xlabel("Time (ms)", fontsize=20)
        ax3.set_ylabel("Adiabaticity Value", fontsize=20)
        ax3.grid()
        ax3.legend()

        fig.suptitle(fr'Time dependence of $\Omega (t)$ and $\Delta (t)$, J = {j}', fontsize=25)
        fig.tight_layout()

        filename = f"{savetext}_j{j}.svg"
        save_figure_in_images(fig, filename)
        plt.show()


def animate_bloch(
    result: Result, 
    times: np.ndarray, 
    duration: float = 0.1, 
    color: str = 'r', 
    animfilename: str = 'anim_gif'
) -> None:
    """
    Creates a Bloch sphere GIF animation from simulated state evolution.

    Parameters
    ----------
    result : qutip.Result
        Simulation results containing states.
    times : np.ndarray
        Time grid.
    duration : float, optional
        Duration of each frame in the GIF.
    color : str, optional
        Color of the Bloch vector.
    animfilename : str, optional
        Target filename for the GIF.
    """
    # Define standard Pauli operators
    sigmax_op = - sigmax()
    sigmay_op = - sigmay()
    sigmaz_op = - sigmaz()

    vectors = []
    pxs, pys, pzs = [], [], []

    for state in result.states:
        # Trace out the motional subsystem to obtain the internal qubit state
        rho = state.ptrace(0)
        
        # Calculate Bloch vector components
        vector = [expect(sigmax_op, rho), expect(sigmay_op, rho), expect(sigmaz_op, rho)]
        vectors.append(vector)
        pxs.append(vector[0])
        pys.append(vector[1])
        pzs.append(vector[2])

    length = len(times)

    # Initialize Bloch sphere visualization object
    b = Bloch()
    b.view = [-40, 30]
    b.vector_color = [color]
    b.point_color = [color]
    b.point_marker = ['o']
    b.point_size = [30]

    images = []
    # Generate frames for the animation
    for i in range(length):
        b.clear()
        b.add_vectors(vectors[i])
        b.add_points([pxs[:i+1], pys[:i+1], pzs[:i+1]], meth='l')
        
        filename = 'temp_file.png'
        b.save(filename)
        images.append(imageio.imread(filename))

    # Compile frames into a GIF
    imageio.mimsave('images/' + animfilename + '.gif', images, duration=duration)