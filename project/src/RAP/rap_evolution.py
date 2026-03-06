"""
Module for Orchestrating Rapid Adiabatic Passage (RAP) Simulations.

This module provides the main simulation loop for executing RAP pulses across 
different molecular rotational manifolds. It handles the initialization of 
density matrices, selection of time-dependent Hamiltonians, execution of the 
QuTiP Master Equation solver, and automated data saving and plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qutip import *
from typing import List, Dict, Tuple, Any, Optional, Union

from RAP.rap_hamiltonians import *
from RAP.rap_operators import *
from RAP.rap_utils import *
from saving import RAP_save_final_state
from RAP.rap_plotting import RAP_plot_internal_and_motional_dynamics


def RAP_simulation(
    dataframe: pd.DataFrame, 
    n_motional: int, 
    b_field_gauss: float, 
    rabi_rate: float, 
    laser_detuning: float, 
    is_minus: bool, 
    times: np.ndarray, 
    T: float, 
    D: float, 
    sigma: float, 
    trap_freq: Optional[float] = None, 
    lamb_dicke: Optional[float] = None, 
    from_simulation: bool = False, 
    dm: int = -1, 
    rabi_flop: Optional[Any] = None, 
    sideband: bool = True, 
    off_resonant: bool = False, 
    init_pop_list: List[float] = [0.5, 0.5],
    j_plot: Optional[int] = None, 
    savetext: str = "rap_pulse", 
    molecule_type: str = "CaH"
) -> Tuple[List[str], Dict[str, Any], Dict[int, float]]:
    """
    Runs a full RAP simulation over all specified rotational manifolds 'j'.

    This function iterates through the transitions defined in the dataframe, 
    constructs the corresponding Hamiltonians and initial density matrices, 
    and solves the temporal evolution using the QuTiP mesolve function.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Contains information about molecular states and transitions.
    n_motional : int
        Number of motional basis states (Fock space dimension).
    b_field_gauss : float
        External magnetic field in Gauss.
    rabi_rate : float
        Peak Rabi frequency of the pulse.
    laser_detuning : float
        Central laser detuning offset.
    is_minus : bool
        Directionality of the transition (determines state mapping).
    times : np.ndarray
        Time array for the evolution simulation.
    T : float
        Total duration of the RAP pulse.
    D : float
        Chirp slope (frequency sweep range) for the detuning.
    sigma : float
        Standard deviation (width) of the Gaussian pulse envelope.
    trap_freq : float, optional
        Motional trap frequency (required for off-resonant carrier terms).
    lamb_dicke : float, optional
        Lamb-Dicke factor for motional coupling.
    from_simulation : bool, optional
        If True, the initial state is derived from prior simulation data.
    dm : int, optional
        Index representing the change in quantum number m. Default is -1.
    rabi_flop : Any, optional
        If provided, a standard Rabi-flop simulation is performed for comparison.
    sideband : bool, optional
        Whether to use Blue Sideband (True) or Carrier (False) Hamiltonians.
    off_resonant : bool, optional
        Whether to include off-resonant carrier interference terms.
    init_pop_list : list of float, optional
        Initial populations for the internal states. Default is [0.5, 0.5].
    j_plot : int, optional
        Specific J manifold to plot in detail.
    savetext : str, optional
        Filename prefix for saving plot images.
    molecule_type : str, optional
        Identifier for the molecular species. Default is "CaH".

    Returns
    -------
    full_paths_list : list of str
        List of absolute filepaths to the saved density matrices.
    args : dict
        The parameters and configuration used for the simulation terms.
    pop_vs_j : dict
        A dictionary mapping the manifold index 'j' to the final excited population.
    """

    # Initialize simulation arguments and rescale dataframe indices
    args, rescaled_df = RAP_args(
        dataframe, is_minus, n_motional, rabi_rate, laser_detuning, 
        times, T, D, sigma, trap_freq=trap_freq, 
        lamb_dicke=lamb_dicke, off_resonant=off_resonant
    )

    # Configuration for the QuTiP master equation solver
    opts = Options(store_states=True, nsteps=80000)

    e_ops = []
    full_paths_list = []
    pop_vs_j = {}

    # Iterate through each transition manifold defined in the arguments
    for term_args in args['terms']:
        j = term_args['j']
        final_time = term_args['final_time']
        print("RAP of the transition in j = ", j)

        # Prepare the initial density matrix (rho)
        rho = RAP_dm(
            term_args, n_motional, from_simulation, 
            sideband=sideband, init_pop_list=init_pop_list
        )
        
        # --- RAP PHASE ---
        if sideband:
            # Use Blue Sideband (BSB) Hamiltonian
            H = RAP_bsb_H_2levels
            print("BSB Hamiltonian created; now simulating")
            result = mesolve(H, rho, times, [], e_ops, args=term_args, options=opts)
        else:
            # Use Carrier Hamiltonian
            H = RAP_carrier_H_2levels
            print("Carrier Hamiltonian created; now simulating")
            result = mesolve(H, rho, times, [], e_ops, args=term_args, options=opts)

        # --- OPTIONAL RABI FLOP PHASE ---
        result_rabiflop = None
        if rabi_flop is not None:
            H_rabiflop = RAP_rabiflop_H_2levels
            print("Hamiltonian created; now simulating")
            result_rabiflop = mesolve(
                H_rabiflop, rho, times, [], e_ops, 
                args=term_args, options=opts
            )

        # Save final state data to disk
        full_path = RAP_save_final_state(
            result, term_args, final_time, b_field_gauss, j, 
            rabi_rate, laser_detuning, RAP=True, 
            from_simulation=from_simulation, dm=dm, molecule_type=molecule_type
        )
        full_paths_list.append(full_path)
        
        # Generate and save dynamics plots
        last_up_pop = RAP_plot_internal_and_motional_dynamics(
            result, times, n_motional, j, dm, 
            result_rf=result_rabiflop, j_plot=j_plot, savetext=savetext
        )

        # Record the final population in the excited state for the J-manifold
        pop_vs_j[j] = last_up_pop
    
    return full_paths_list, args, pop_vs_j















