# File: saving.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Module for saving simulation results and figures for molecular pumping and RAP.

This module provides functions to:
- Save matplotlib figures in the nearest 'images' directory.
- Save final quantum states of pumping simulations.
- Save final quantum states of RAP simulations.
- Save RAP signature data.
- Compute and save populations of final states for plotting.
"""

from pathlib import Path
import os
import joblib
import numpy as np
from qutip import tensor, basis, qeye, expect


def save_figure_in_images(fig, filename: str = "figure.png"):
    """
    Save a matplotlib figure inside the nearest 'images' directory found
    by traversing upwards in the directory tree.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str, optional
        Name of the file to save (default is "figure.png").

    Raises
    ------
    FileNotFoundError
        If no 'images' directory is found in the current or parent directories.
    """
    current_path = Path.cwd()

    # Search in current directory and parents
    for parent in [current_path] + list(current_path.parents):
        images_path = parent / "images"
        if images_path.is_dir():
            # Save figure with tight bounding box
            fig.savefig(images_path / filename, bbox_inches="tight")
            print(f"Saved figure in: {images_path / filename}")
            return

    raise FileNotFoundError("No 'images' directory found in current or parent directories.")


def save_final_state(result, 
                     args, 
                     final_time, 
                     b_field, 
                     j_val, 
                     rr, 
                     las_det, 
                     cr, 
                     molecule_type, 
                     last_pulse=False):
    """
    Save the final quantum state of a simulation and relevant parameters
    for a single pumping process.

    Parameters
    ----------
    result : qutip.Result
        Result object containing the full time evolution of the system.
    args : dict
        Simulation arguments or pulse parameters.
    final_time : float
        Final time of the simulation.
    b_field : float
        Magnetic field in Gauss.
    j_val : float
        Total angular momentum quantum number.
    rr : float
        Rabi rate.
    las_det : float
        Laser detuning.
    cr : float
        Cooling rate.
    molecule_type : str
        Molecule name, used for folder naming.
    last_pulse : bool, optional
        If True, saves in 'pumping_RAP_data' folder; otherwise in 'results_pumping'. Default is False.

    Returns
    -------
    str
        Full path to the saved .pkl file.
    """
    # Extract final density matrix
    rho_final = result.states[-1]

    # Determine folder based on last_pulse flag
    current_path = Path(__file__).resolve()
    if not last_pulse:
        target_folder_name = f"results_pumping_{molecule_type.lower()}"
    else:
        target_folder_name = f"pumping_RAP_data_{molecule_type.lower()}"

    save_dir = None

    # Search in parent directories
    for parent in current_path.parents:
        candidate = parent / target_folder_name
        if candidate.exists() and candidate.is_dir():
            save_dir = candidate
            break

    # If not found, create in project root
    if save_dir is None:
        project_root = current_path.parents[-2]
        save_dir = project_root / target_folder_name
        save_dir.mkdir(exist_ok=True)

    # Base filename with all relevant parameters
    base_filename = (
        f"J{j_val}_B{b_field:.2f}G_T{final_time/1000}ms_"
        f"RR{rr*1000/(2*np.pi):.4f}_DET{las_det*1000/(2*np.pi):.4f}_"
        f"CR{cr*1000/(2*np.pi):.4f}"
    )

    # Incremental suffix to avoid overwriting
    i = 0
    while True:
        filename = f"{base_filename}_{i}.pkl"
        full_path = os.path.join(save_dir, filename)
        if not os.path.exists(full_path):
            break
        i += 1

    # Data dictionary to save
    data_to_save = {
        "rho_final": rho_final,
        "args": args,
        "final_time": final_time,
        "b_field_gauss": b_field,
        "j_val": j_val,
        "rabi_rate": rr/(2*np.pi),
        "laser_detuning": las_det/(2*np.pi),
        "cooling_rate": cr/(2*np.pi)
    }

    joblib.dump(data_to_save, full_path)
    print(f"Final state saved in: {full_path}")
    return full_path



def RAP_save_final_state(result, 
                         args, 
                         final_time, 
                         b_field, 
                         j_val, 
                         rr, 
                         las_det, 
                         RAP, 
                         from_simulation, 
                         dm, 
                         molecule_type):
    """
    Save the final state and simulation parameters for a RAP (Rapid Adiabatic Passage) process.

    Parameters
    ----------
    result : qutip.Result
        Full time evolution result object.
    args : dict
        Dictionary with pulse shape and system parameters (T, D, sigma, etc.).
    final_time : float
        Final time of the simulation.
    b_field : float
        Magnetic field in Gauss.
    j_val : float
        Total angular momentum quantum number.
    rr : float
        Rabi frequency.
    las_det : float
        Laser detuning.
    RAP : bool
        True if RAP was applied.
    from_simulation : bool
        True if the result comes from a simulation.
    dm : int
        Delta-m quantum number for RAP branch.
    molecule_type : str
        Molecule name for folder naming.

    Returns
    -------
    str
        Full path to the saved .pkl file.
    """
    rho_final = result.states[-1]

    T = args['T']
    D = args['D']
    sigma = args['sigma']

    current_path = Path(__file__).resolve()
    target_folder_name = f"results_rap_{molecule_type.lower()}"
    save_dir = None

    # Search parent directories
    for parent in current_path.parents:
        candidate = parent / target_folder_name
        if candidate.exists() and candidate.is_dir():
            save_dir = candidate
            break

    # Create folder if not found
    if save_dir is None:
        project_root = current_path.parents[-2]
        save_dir = project_root / target_folder_name
        save_dir.mkdir(exist_ok=True)

    # Data dictionary for saving
    data_to_save = {
        "rho_final": rho_final,
        "args": args,
        "final_time": final_time,
        "b_field_gauss": b_field,
        "j_val": j_val,
        "rabi_rate": rr/(2*np.pi),
        "laser_detuning": las_det/(2*np.pi),
        "RAP": RAP,
        "from_simulation": from_simulation,
        "dm": dm
    }

    # Base filename including pulse parameters
    base_filename = f"RAP_J{j_val}_B{b_field:.2f}G_T{T/1000:.2f}ms_SIGMA{sigma/1000:.2f}ms_{D/(2*np.pi):.4f}_RR{rr*1000/(2*np.pi):.4f}_DET{las_det*1000/(2*np.pi):.4f}_DM{dm}_fromsim{from_simulation}.pkl"

    # Incremental suffix to avoid overwriting
    i = 0
    while True:
        filename = f"{base_filename}_{i}.pkl"
        full_path = os.path.join(save_dir, filename)
        if not os.path.exists(full_path):
            break
        i += 1

    joblib.dump(data_to_save, full_path)
    print(f"Final state saved in: {full_path}")
    return full_path


def save_rap_signature(j_vals, pop_vals, filename="rap_pulse.pkl", molecule_type="cah"):
    """
    Save the RAP signature data in a dedicated folder.

    Parameters
    ----------
    j_vals : list or np.ndarray
        J manifold values.
    pop_vals : list or np.ndarray
        Corresponding populations for the RAP signature.
    filename : str, optional
        Name of the file to save (default is "rap_pulse.pkl").
    molecule_type : str, optional
        Molecule type used for folder naming (default "cah").

    Returns
    -------
    str
        Full path to the saved .pkl file.

    Raises
    ------
    FileNotFoundError
        If the target folder is not found in the directory tree.
    """
    current_path = Path(__file__).resolve()
    target_folder_name = f"pumping_RAP_data_{molecule_type.lower()}"
    save_dir = None

    # Search parent directories
    for parent in current_path.parents:
        candidate = parent / target_folder_name
        if candidate.exists() and candidate.is_dir():
            save_dir = candidate
            break

    if save_dir is None:
        raise FileNotFoundError(
            f"Folder '{target_folder_name}' not found in current directory tree."
        )

    full_path = save_dir / filename

    data_to_save = {
        "j_vals": j_vals,
        "pop_vals": pop_vals
    }

    joblib.dump(data_to_save, full_path)
    print(f"RAP signature saved in: {full_path}")
    return str(full_path)



def plot_final_state_pop(results, 
                         mo1, 
                         init_state, 
                         idx_meas, 
                         idx_manifold, 
                         is_minus,
                         n_internal, 
                         n_motional, 
                         rabi_rate, 
                         target_folder_name, 
                         t_value=None):
    """
    Compute and save the populations of the internal states at the final
    or specified time of the simulation.

    Parameters
    ----------
    results : qutip.Result
        Simulation result containing state evolution.
    mo1 : any
        Placeholder for molecule object.
    init_state : int
        Index of initial state.
    idx_meas : int
        Measurement index.
    idx_manifold : int
        J manifold index.
    is_minus : bool
        Flag for selection of directionality branch.
    n_internal : int
        Number of internal states.
    n_motional : int
        Number of motional states.
    rabi_rate : float
        Rabi rate in Hz.
    target_folder_name : str
        Folder name for saving the .pkl file.
    t_value : float, optional
        Specific time to plot; if None, final state is used.

    Returns
    -------
    str
        Full path to the saved .pkl file.
    """
    # Select final or specified state
    if t_value is None:
        fin = results.states[-1]
    else:
        idx_marker = np.argmin(np.abs(np.array(results.times) - t_value))
        fin = results.states[idx_marker]

    # Compute populations for all internal states
    populations = np.zeros(n_internal)
    for j in range(n_internal):
        Pj_op = tensor(basis(n_internal, j) * basis(n_internal, j).dag(), qeye(n_motional))
        populations[j] = expect(Pj_op, fin)

    current_path = Path.cwd().resolve()
    print("Current path for saving final state:", current_path)
    save_dir = None

    # Check if target folder exists in current or parent directories
    candidate = current_path / target_folder_name
    if candidate.exists() and candidate.is_dir():
        save_dir = candidate
    else:
        for parent in current_path.parents:
            candidate = parent / target_folder_name
            if candidate.exists() and candidate.is_dir():
                save_dir = candidate
                break

    # Create folder if not found
    if save_dir is None:
        project_root = current_path.parents[-2]
        save_dir = project_root / target_folder_name
        save_dir.mkdir(exist_ok=True)

    # Base filename with parameters
    base_filename = (
        f"off_init{init_state}_meas{idx_meas}_J{idx_manifold}_ismin{is_minus}_rr{rabi_rate/(2*np.pi):.4f}"
    )

    # Incremental suffix to avoid overwriting
    i = 0
    while True:
        filename = f"{base_filename}_{i}.pkl"
        full_path = os.path.join(save_dir, filename)
        if not os.path.exists(full_path):
            break
        i += 1

    # Data dictionary to save
    data_to_save = {
        "index_meas": idx_meas,
        "index_manifold": idx_manifold,
        "init_state": init_state,
        "t_value": t_value,
        "is_minus": is_minus,
        "populations": populations,
        "n_internal": n_internal,
        "rho_final_t_value": fin,
        "rabi_rate": rabi_rate
    }

    joblib.dump(data_to_save, full_path)
    print(f"Final state saved in: {full_path}")
    return full_path