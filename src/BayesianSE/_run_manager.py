"""
Management Module for Bayesian Simulation Runs and Data Logging.

This module provides utilities to organize and store data from Bayesian state 
estimation experiments. It handles:
- Automatic creation of numbered run directories.
- Exporting of matplotlib figures to the current run folder.
- Logging of comprehensive metadata (experimental and simulation parameters).
- Saving numerical results in JSON format.
"""

from pathlib import Path
import re
import json
from typing import Optional, Dict, Any, Union

CURRENT_RUN_PATH: Optional[Path] = None


def init_new_run() -> Path:
    """
    Initializes a new run directory inside the nearest 'bayesian_runs' folder.

    The function climbs up the directory tree to find 'bayesian_runs', identifies 
    the highest existing run number, and creates the 
    subsequent folder.

    Returns
    -------
    Path
        The absolute path to the newly created run directory.

    Raises
    ------
    FileNotFoundError
        If no 'bayesian_runs' directory is found in the path hierarchy.
    """
    global CURRENT_RUN_PATH

    current_path = Path.cwd()

    for parent in [current_path] + list(current_path.parents):
        images_path = parent / "bayesian_runs"
        if images_path.is_dir():

            run_dirs = []
            for p in images_path.iterdir():
                m = re.fullmatch(r"bayesian_run_(\d+)", p.name)
                if p.is_dir() and m:
                    run_number = int(m.group(1))
                    run_dirs.append(run_number)

                    try:
                        run_number_alt = int(p.name.split("_")[1])
                        run_dirs.append(run_number_alt)
                    except (ValueError, IndexError):
                        pass

            next_run = max(run_dirs) + 1 if run_dirs else 1

            run_path = images_path / f"bayesian_run_{next_run}"
            run_path.mkdir(exist_ok=False)

            CURRENT_RUN_PATH = run_path
            print(f"Initialized new run folder: {run_path}")
            return run_path

    raise FileNotFoundError("No 'bayesian_runs' directory found in parent tree.")


def plot_bayesian_run(fig: Any, filename: str) -> None:
    """
    Saves a matplotlib figure inside the current run directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to be saved.
    filename : str
        The name of the file (e.g., 'convergence_plot.png').

    Raises
    ------
    RuntimeError
        If the run folder has not been initialized via init_new_run().
    """
    global CURRENT_RUN_PATH

    if CURRENT_RUN_PATH is None:
        raise RuntimeError("Run folder not initialized. Call init_new_run() first.")

    save_path = CURRENT_RUN_PATH / filename
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved figure in: {save_path}")


def save_metadata(
    molecule_type: str,
    temperature: float,
    b_field_gauss: float,
    j_max: int,
    rabi_by_j: Union[float, Dict],
    dephased: bool,
    coherence_time_us: float,
    is_minus: bool,
    false_positive_rate: float,
    false_negative_rate: float,
    noise_params: Optional[Dict],
    seed: Optional[int],
    laser_miscalibration: Optional[Dict],
    seed_miscalibration: Optional[int],
    noise_params_estim: Optional[Dict],
    laser_miscalibration_estim: Optional[Dict],
    pop_fit: Any,
    N: int,
    num_updates: int,
    block_steps: int,
    type_block: Optional[str],
    apply_pumping: bool,
    marginalization: bool,
    false_rates: bool,
    save_data: bool,
    only_total: bool,
    max_excitation: float
) -> None:
    """
    Logs all experimental and simulation parameters into a metadata text file.

    Parameters
    ----------
    molecule_type : str
        Molecular species identifier.
    temperature : float
        Rotational temperature.
    b_field_gauss : float
        Magnetic field strength.
    j_max : int
        Maximum rotational manifold.
    rabi_by_j : float or dict
        Rabi rate configuration.
    dephased : bool
        If dephasing was included in the model.
    coherence_time_us : float
        Coherence time in microseconds.
    is_minus : bool
        Pulse direction.
    false_positive_rate : float
        FPR of the detector.
    false_negative_rate : float
        FNR of the detector.
    noise_params : dict
        Physical noise settings for the simulator.
    seed : int
        Simulation random seed.
    laser_miscalibration : dict
        Systematic errors for the simulator.
    seed_miscalibration : int
        Seed for calibration errors.
    noise_params_estim : dict
        Noise model used by the estimator.
    laser_miscalibration_estim : dict
        Miscalibration model used by the estimator.
    pop_fit : Any
        Pumping efficiency fits.
    N : int
        Motional states cutoff.
    num_updates : int
        Number of global sweeps.
    block_steps : int
        Steps per measurement block.
    type_block : str
        The block scheduling pattern.
    apply_pumping : bool
        If repumping logic was active.
    marginalization : bool
        If the estimator marginalized over noise.
    false_rates : bool
        If FPR/FNR were considered in the likelihood.
    save_data : bool
        If history data was recorded.
    only_total : bool
        Plotting flag.
    max_excitation : float
        Pulse excitation limit.
    """
    global CURRENT_RUN_PATH

    if CURRENT_RUN_PATH is None:
        raise RuntimeError("Run folder not initialized. Call init_new_run() first.")

    metadata_path = CURRENT_RUN_PATH / "metadata.txt"

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
        "only_total": only_total,
        "max_excitation": max_excitation
    }

    with open(metadata_path, "w") as f:
        f.write("### Bayesian State Estimation Run Metadata ###\n\n")
        for key, value in inputs.items():
            f.write(f"{key}: {value}\n")

    print(f"Run metadata saved at: {metadata_path}")


def save_results(results: Dict[str, Any], filename: str = "results.json") -> None:
    """
    Saves the final results dictionary in JSON format inside the run folder.

    Parameters
    ----------
    results : dict
        The numerical results and history to be exported.
    filename : str, optional
        The JSON filename. Default is "results.json".

    Raises
    ------
    RuntimeError
        If the run folder has not been initialized.
    TypeError
        If the results contain non-serializable objects like numpy arrays.
    """
    global CURRENT_RUN_PATH

    if CURRENT_RUN_PATH is None:
        raise RuntimeError("Run folder not initialized. Call init_new_run() first.")

    results_path = CURRENT_RUN_PATH / filename

    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
    except TypeError as e:
        # Inform user about numpy conversion if serialization fails
        raise TypeError(
            "Results dictionary contains non-JSON-serializable objects. "
            "Convert numpy arrays to lists or floats before saving."
        ) from e

    print(f"Run results saved at: {results_path}")