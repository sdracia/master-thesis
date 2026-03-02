from pathlib import Path
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, basis, qeye, expect



####################################################################
################## SAVE IMAGES FUNCTION ############################
####################################################################



def save_figure_in_images(fig, filename: str = "figure.png"):
    """
    Save a matplotlib figure inside the nearest 'images' directory found by going up the directory tree.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str
        Name of the file to save (e.g. "plot.svg").
    """
    current_path = Path.cwd()
    
    for parent in [current_path] + list(current_path.parents):
        images_path = parent / "images"
        if images_path.is_dir():
            fig.savefig(images_path / filename, bbox_inches="tight")
            print(f"Saved figure in: {images_path / filename}")
            return
    
    raise FileNotFoundError("No 'images' directory found in current or parent directories.")


#############################################################
################### SAVE FINAL STATE FUNCTION ###############
#############################################################


def save_final_state(result, args, final_time, b_field, j_val, rr, las_det, cr, molecule_type, last_pulse = False):
    rho_final = result.states[-1]


    current_path = Path(__file__).resolve()
    if not last_pulse:
        target_folder_name = f"results_pumping_{molecule_type.lower()}"
    else:
        target_folder_name = f"pumping_RAP_data_{molecule_type.lower()}"

    save_dir = None

    # Search going backwards
    for parent in current_path.parents:
        candidate = parent / target_folder_name
        if candidate.exists() and candidate.is_dir():
            save_dir = candidate
            break

    # If not found, create it in project root
    if save_dir is None:
        project_root = current_path.parents[-2]
        save_dir = project_root / target_folder_name
        save_dir.mkdir(exist_ok=True)


    base_filename = (
        f"J{j_val}_B{b_field:.2f}G_T{final_time/1000}ms_"
        f"RR{rr*1000/(2*np.pi):.4f}_DET{las_det*1000/(2*np.pi):.4f}_"
        f"CR{cr*1000/(2*np.pi):.4f}"
    )
    
    # Find a non-existing filename with incremental suffix
    i = 0
    while True:
        filename = f"{base_filename}_{i}.pkl"
        full_path = os.path.join(save_dir, filename)
        if not os.path.exists(full_path):
            break
        i += 1

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


#####################################################################
################## SAVE FINAL STATE FUNCTION FOR RAP ###############
#####################################################################


def RAP_save_final_state(result, args, final_time, b_field, j_val, rr, las_det, RAP, from_simulation, dm, molecule_type):
    """
    -------------------------------------------------------------------------------
     Save the final state and relevant simulation parameters for a RAP process.

     Parameters
     ----------
     result : qutip.Result
         Result object containing the full time evolution of the system.
     args : dict
         Dictionary containing the pulse shape and system parameters (T, D, sigma, etc.).
     final_time : float
         Final time of the simulation (used as timestamp for output file).
     b_field : float
         Magnetic field in Gauss used during the simulation.
     j_val : float
         Total angular momentum quantum number.
     rr : float
         Rabi frequency in Hz (will be stored in kHz units).
     las_det : float
         Laser detuning in Hz (will be stored in kHz units).
     RAP : bool
         Flag indicating whether RAP was used.
     from_simulation : bool
         Flag indicating whether this result was from a full simulation (as opposed to analytical or approximate methods).
     dm : int
         Δm quantum number identifying the RAP branch.

     Returns
     -------
     full_path : str
         Path to the saved .pkl file containing final state and metadata.
    -------------------------------------------------------------------------------
    """

    rho_final = result.states[-1]
    
    T = args['T']
    D = args['D']
    sigma = args['sigma']



    current_path = Path(__file__).resolve()
    target_folder_name = f"results_rap_{molecule_type.lower()}"

    save_dir = None

    # Search going backwards
    for parent in current_path.parents:
        candidate = parent / target_folder_name
        if candidate.exists() and candidate.is_dir():
            save_dir = candidate
            break

    # If not found, create it in project root
    if save_dir is None:
        project_root = current_path.parents[-2]
        save_dir = project_root / target_folder_name
        save_dir.mkdir(exist_ok=True)



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

    base_filename = f"RAP_J{j_val}_B{b_field:.2f}G_T{T/1000:.2f}ms_SIGMA{sigma/1000:.2f}ms_{D/(2*np.pi):.4f}_RR{rr*1000/(2*np.pi):.4f}_DET{las_det*1000/(2*np.pi):.4f}_DM{dm}_fromsim{from_simulation}.pkl"

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






def save_rap_signature(j_vals, pop_vals, filename = "rap_pulse.pkl", molecule_type="cah"):
    """
    Save RAP signature data inside the folder:
        final_RAP_pulses_<molecule_type>

    The folder is searched going backwards in the directory tree.
    If not found, a FileNotFoundError is raised.
    """

    current_path = Path(__file__).resolve()
    target_folder_name = f"pumping_RAP_data_{molecule_type.lower()}"

    save_dir = None

    # ---------------------------------------------------
    # Search going backwards in directory tree
    # ---------------------------------------------------
    for parent in current_path.parents:
        candidate = parent / target_folder_name
        if candidate.exists() and candidate.is_dir():
            save_dir = candidate
            break

    if save_dir is None:
        raise FileNotFoundError(
            f"Folder '{target_folder_name}' not found "
            f"in current directory tree."
        )


    full_path = save_dir / filename

    data_to_save = {
        "j_vals": j_vals,
        "pop_vals": pop_vals
    }

    joblib.dump(data_to_save, full_path)

    print(f"RAP signature saved in: {full_path}")

    return str(full_path)




def plot_final_state_pop(results, mo1, init_state, idx_meas, idx_manifold, is_minus, n_internal, n_motional, rabi_rate, target_folder_name, t_value=None):
    if t_value is None:
        fin = results.states[-1]
    else:
        idx_marker = np.argmin(np.abs(np.array(results.times) - t_value))
        fin = results.states[idx_marker]

    populations = np.zeros(n_internal)
                        
    for j in range(n_internal):
        Pj_op = tensor(basis(n_internal, j) * basis(n_internal, j).dag(), qeye(n_motional))
        populations[j] = expect(Pj_op, fin)


    # plt.figure(figsize=(5, 3))

    # plt.bar(range(n_internal), populations, color='tab:blue', alpha=0.7)
    # plt.title(f"Final state population at {t_value:.2f}")
    # plt.xlabel('Internal state index')
    # plt.ylabel('Population')
    # plt.xticks(range(n_internal))
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    current_path = Path.cwd().resolve()
    print("Current path for saving final state:", current_path)
    save_dir = None

    candidate = current_path / target_folder_name
    if candidate.exists() and candidate.is_dir():
        save_dir = candidate
    else:
        # 2️⃣ Poi eventualmente cerca nei parent
        for parent in current_path.parents:
            candidate = parent / target_folder_name
            if candidate.exists() and candidate.is_dir():
                save_dir = candidate
                break


    # If not found, create it in project root
    if save_dir is None:
        project_root = current_path.parents[-2]
        save_dir = project_root / target_folder_name
        save_dir.mkdir(exist_ok=True)


    base_filename = (
        f"off_init{init_state}_meas{idx_meas}_J{idx_manifold}_ismin{is_minus}_rr{rabi_rate/(2*np.pi):.4f}"
    )
    
    # Find a non-existing filename with incremental suffix
    i = 0
    while True:
        filename = f"{base_filename}_{i}.pkl"
        full_path = os.path.join(save_dir, filename)
        if not os.path.exists(full_path):
            break
        i += 1

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

    # directory = os.path.dirname(full_path)
    # base_name = os.path.splitext(os.path.basename(full_path))[0]
    # image_path = os.path.join(directory, f"{base_name}.svg")

    # plt.savefig(image_path)
    # print(f"Image saved as: {image_path}")



    return full_path