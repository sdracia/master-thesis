# File: pump_evolution.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Main Simulation Orchestrator for Molecular Pumping and Cooling.

This module provides the core execution logic for simulating a sequence of 
laser-driven pumping stages on a molecular system. It integrates Hilbert space 
initialization, time-dependent Hamiltonian construction, and Master Equation 
evolution using QuTiP.
"""

import numpy as np
from qutip import *
from typing import List, Dict, Any, Optional, Tuple

from pumping.pump_utils import *
from saving import save_final_state
from pumping.pump_plotting import plot_internal_and_motional_dynamics
from pumping.pump_hamiltonians import H_bsb_manifold
from pumping.pump_operators import collapse_cooling_op


def run_pumping(
    b_field_gauss: float,
    j_max: int,
    temperature: float,
    j_val: int,
    n_motional: int,
    laser_configs: List[Dict[str, Any]],
    e_ops: List[Qobj],
    molecule_type: str = "CaH",
    save_intermediate_states: bool = False,
    plot_intermediate_dynamics: bool = False
) -> None:
    """
    Executes a sequence of pumping stages on a molecule-motion system.

    Parameters
    ----------
    b_field_gauss : float
        Magnetic field strength in Gauss.
    j_max : int
        Maximum rotational quantum number considered for the molecule.
    temperature : float
        Initial temperature of the molecular sample.
    j_val : int
        The specific rotational manifold J where the pumping is applied.
    n_motional : int
        Number of Fock states in the motional Hilbert space (cutoff).
    laser_configs : list of dict
        A list where each dictionary defines parameters for a specific pulse:
        (is_minus, times, laser_detuning, rabi_rate, raman_config).
    e_ops : list of Qobj
        List of expectation operators to monitor during the evolution.
    molecule_type : str, optional
        The molecular species identifier. Default is "CaH".
    save_intermediate_states : bool, optional
        If True, exports the density matrix after each pulse. Default is False.
    plot_intermediate_dynamics : bool, optional
        If True, generates plots for each step in the sequence. Default is False.

    Returns
    -------
    None
    """
    
    # 1. Initialization and Validation
    validate_laser_configs(laser_configs)

    mo1, states1, mo1_dm2, states1_dm2 = initialize_molecule(
        molecule_type,
        b_field_gauss,
        j_max,
        temperature
    )

    transitions_in_j = mo1.transition_df[mo1.transition_df["j"] == j_val]
    transitions_in_j_dm2 = mo1_dm2.transition_df[mo1_dm2.transition_df["j"] == j_val]

    n_internal = 2 * (2 * j_val + 1)

    if j_val > 8:
        keep_sub_manifold_levels = 5
        transitions_in_j = cut_trans_df(transitions_in_j, j_val, keep_sub_manifold_levels)
        transitions_in_j_dm2 = cut_trans_df(transitions_in_j_dm2, j_val, keep_sub_manifold_levels)

        sub_index = min(keep_sub_manifold_levels, 2 * j_val + 1) 
        n_internal = 2 * sub_index  

    # 2. Initial State Preparation
    states = [basis(n_internal, i) for i in range(n_internal)]
    rho_internal = sum([ket2dm(state) for state in states]) / n_internal

    index_motional = 0
    assert 0 <= index_motional < n_motional, "index_motional out of range"
    psi_motional = basis(n_motional, index_motional)
    rho_motional = ket2dm(psi_motional)

    rho0 = tensor(rho_internal, rho_motional)

    # 3. Simulation Configuration
    opts = Options(store_states=True, progress_bar="text", nsteps=20000)
    results = []
    
    scheme_id = "".join([pulse["raman_config"][-1] for pulse in laser_configs])

    # 4. Sequential Evolution (Pulse Loop)
    for i, config in enumerate(laser_configs):

        is_minus = config["is_minus"]
        times = config["times"]
        laser_detuning = config["laser_detuning"]
        rabi_rate = config["rabi_rate"]
        manifold_type = config["raman_config"]

        if manifold_type == "dm2":
            transitions_selected = transitions_in_j_dm2
        else:
            transitions_selected = transitions_in_j

        cooling_rate = rabi_rate
        final_time = times[-1]

        # Setup operators for the Master Equation
        c_ops = collapse_cooling_op(cooling_rate, n_internal, n_motional)
        H_tot, args = H_bsb_manifold(
            transitions_selected, j_val, is_minus, 
            n_motional, n_internal, rabi_rate, laser_detuning
        )

        result = mesolve(H_tot, rho0, times, c_ops, e_ops, args=args, options=opts)

        if save_intermediate_states:
            full_path = save_final_state(
                result, args, final_time, b_field_gauss, 
                j_val, rabi_rate, laser_detuning, cooling_rate, molecule_type
            )
            # Mark the file as the last pulse if applicable
            if i == len(laser_configs) - 1:
                full_path = save_final_state(
                    result, args, final_time, b_field_gauss, j_val, 
                    rabi_rate, laser_detuning, cooling_rate, molecule_type, 
                    last_pulse=True
                )

        # Generate intermediate visualization
        if plot_intermediate_dynamics:
            savetext = (
                 f"{molecule_type}_j{j_val}_pulse{i+1}_"
                 f"{len(laser_configs)}stg_{scheme_id}.svg"
            )
            plot_internal_and_motional_dynamics(
                result, times, n_internal, n_motional, 
                title="Population evolution", savetext=savetext, only_pop=False
            )

        # Update the initial state for the next pulse with the current final state
        rho0 = result.states[-1]
        results.append([result, times])

    # 5. Final Visualization
    savetext = (
        f"{molecule_type}_j{j_val}_"
        f"{len(laser_configs)}stg_{scheme_id}.svg"
    )

    plot_internal_and_motional_dynamics(
        results, 
        times=None, 
        n_internal=n_internal, 
        n_motional=n_motional, 
        title="Population evolution", 
        savetext=savetext, 
        only_pop=True
    )