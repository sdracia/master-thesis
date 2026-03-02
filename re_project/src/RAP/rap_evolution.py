import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qutip import *

from RAP.rap_hamiltonians import *
from RAP.rap_operators import *
from RAP.rap_utils import *
from saving import RAP_save_final_state
from RAP.rap_plotting import RAP_plot_internal_and_motional_dynamics




def RAP_simulation(dataframe, 
                   n_motional, b_field_gauss, rabi_rate, laser_detuning, is_minus, times, 
                   T, D, sigma, 
                   trap_freq = None, lamb_dicke = None, 
                   from_simulation = False, dm = -1, rabi_flop = None, 
                   sideband = True, off_resonant = False, 
                   init_pop_list = [0.5, 0.5],
                   j_plot = None, savetext = "rap_pulse", molecule_type = "CaH"):
    """
    > Runs a full RAP simulation over all specified manifolds `j` using `mesolve`.

    Builds Hamiltonians, initial states, and solves the Schrödinger equation or Lindblad master equation (if collapse operators added).
    Optionally computes a second Rabi-flop evolution and saves final states and dynamics plots.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        Contains information about molecular states and transitions.
    n_motional : int
        Number of motional basis states.
    b_field_gauss : float
        Magnetic field.
    rabi_rate : float
        Peak Rabi frequency.
    laser_detuning : float
        Central laser detuning.
    is_minus : bool
        Directionality of the transition.
    times : array
        Time array for evolution.
    T : float
        Total duration of the RAP pulse.
    D : float
        Chirp slope for detuning sweep.
    sigma : float
        Width of the Gaussian pulse.
    trap_freq : float, optional
        Frequency of the trap (used in off-resonant terms).
    lamb_dicke : float, optional
        Lamb-Dicke factor.
    from_simulation : bool, default=False
        If True, initial states come from prior simulations.
    dm : int, default=-1
        Index of the varation of quantum number m.
    rabi_flop : any, optional
        If provided, a Rabi-flop simulation is also performed.
    sideband : bool, default=True
        Whether sideband coupling is active.
    off_resonant : bool, default=False
        Whether to include off-resonant carrier terms.

    Returns:
    --------
    full_paths_list : list of str
        Filepaths to saved density matrices.
    args : dict
        Dictionary of RAP term parameters for all transitions.
    """

    args, rescaled_df = RAP_args(dataframe, is_minus, n_motional, rabi_rate, laser_detuning, times, T, D, sigma, trap_freq=trap_freq, lamb_dicke=lamb_dicke, off_resonant=off_resonant)

    # opts = Options(store_states=True, progress_bar="text", nsteps=80000)
    opts = Options(store_states=True, nsteps=80000)


    e_ops = []

    full_paths_list = []

    pop_vs_j = {}

    for term_args in args['terms']:
        j = term_args['j']
        final_time = term_args['final_time']
        print("RAP of the transition in j = ", j)

        rho = RAP_dm(term_args, n_motional, from_simulation, sideband = sideband, init_pop_list=init_pop_list)
        
        # RAP
        if sideband:    # bsb
            H = RAP_bsb_H_2levels
            print("BSB Hamiltonian created; now simulating")
            result = mesolve(H, rho, times, [], e_ops, args=term_args, options=opts)
        else:   # carrier
            H = RAP_carrier_H_2levels
            print("Carrier Hamiltonian created; now simulating")
            result = mesolve(H, rho, times, [], e_ops, args=term_args, options=opts)

        result_rabiflop = None
        if rabi_flop is not None:
            # RABI FLOP
            H_rabiflop = RAP_rabiflop_H_2levels
            print("Hamiltonian created; now simulating")
            result_rabiflop = mesolve(H_rabiflop, rho, times, [], e_ops, args=term_args, options=opts)

        full_path = RAP_save_final_state(result, term_args, final_time, b_field_gauss, j, rabi_rate, laser_detuning, RAP = True, from_simulation = from_simulation, dm = dm, molecule_type = molecule_type)
        full_paths_list.append(full_path)
        
        
        last_up_pop = RAP_plot_internal_and_motional_dynamics(result, times, n_motional, j, dm, result_rf=result_rabiflop, j_plot = j_plot, savetext = savetext)

        pop_vs_j[j] = last_up_pop
    

    return full_paths_list, args, pop_vs_j



