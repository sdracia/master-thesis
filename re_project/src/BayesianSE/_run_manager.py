from pathlib import Path
import re

CURRENT_RUN_PATH = None

def init_new_run():
    """
    Initialize a new run directory inside the nearest 'bayesian_runs' folder.
    Creates 'run_X' where X is the next available number and stores it in CURRENT_RUN_PATH.
    """
    global CURRENT_RUN_PATH

    current_path = Path.cwd()

    for parent in [current_path] + list(current_path.parents):
        images_path = parent / "bayesian_runs"
        if images_path.is_dir():

            # trova tutte le cartelle run_N
            run_dirs = []
            for p in images_path.iterdir():
                m = re.fullmatch(r"bayesian_run_(\d+)", p.name)
                if p.is_dir() and m:
                    run_number = int(m.group(1))
                    run_dirs.append(run_number)

                    try:
                        run_number = int(p.name.split("_")[1])
                        run_dirs.append(run_number)
                    except ValueError:
                        pass
            # determina la prossima run
            next_run = max(run_dirs) + 1 if run_dirs else 1

            # path nuova run
            run_path = images_path / f"bayesian_run_{next_run}"
            run_path.mkdir(exist_ok=False)

            CURRENT_RUN_PATH = run_path
            print(f"Initialized new run folder: {run_path}")
            return run_path

    raise FileNotFoundError("No 'bayesian_runs' directory found.")



def plot_bayesian_run(fig, filename):
    """
    Save the figure inside the CURRENT_RUN_PATH.
    You MUST call init_new_run() before using this function.
    """
    global CURRENT_RUN_PATH

    if CURRENT_RUN_PATH is None:
        raise RuntimeError("Run folder not initialized. Call init_new_run() first.")

    save_path = CURRENT_RUN_PATH / filename
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved figure in: {save_path}")



def save_metadata(molecule_type, temperature, b_field_gauss, j_max, rabi_by_j, dephased, coherence_time_us, is_minus,
                  false_positive_rate, false_negative_rate, noise_params, seed, laser_miscalibration, seed_miscalibration,
                  noise_params_estim, laser_miscalibration_estim, pop_fit, N, num_updates,
                  block_steps, type_block, apply_pumping, marginalization, false_rates, save_data, only_total):
    
    global CURRENT_RUN_PATH

    if CURRENT_RUN_PATH is None:
        raise RuntimeError("Run folder not initialized. Call init_new_run() first.")

    metadata_path = CURRENT_RUN_PATH / "metadata.txt"

    with open(metadata_path, "w") as f:
        f.write("### Bayesian State Estimation Run Metadata ###\n\n")
        
        inputs = {
            "molecule_type": molecule_type,
            "temperature": temperature,
            "b_field_gauss": b_field_gauss,
            "j_max": j_max,
            "rabi_by_j": rabi_by_j,
            "dephased": dephased,
            "coherence_time_us": coherence_time_us,
            "is_minus": is_minus,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "noise_params": noise_params,
            "seed": seed,
            "laser_miscalibration": laser_miscalibration,
            "seed_miscalibration": seed_miscalibration,
            "noise_params_estim": noise_params_estim,
            "laser_miscalibration_estim": laser_miscalibration_estim,
            "pop_fit": pop_fit,
            "N": N,
            "num_updates": num_updates,
            "block_steps": block_steps,
            "type_block": type_block,
            "apply_pumping": apply_pumping,
            "marginalization": marginalization,
            "false_rates": false_rates,
            "save_data": save_data,
            "only_total": only_total
        }

        for key, value in inputs.items():
            f.write(f"{key}: {value}\n")

    print(f"Run metadata saved at: {metadata_path}")
