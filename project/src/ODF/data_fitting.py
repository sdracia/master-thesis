"""
Module for ODF experimental data retrieval and signal fitting.

This module provides utility functions to:
- Define a sinc-squared mathematical model for spectral fitting.
- Automatically locate experimental data files within the project structure.
- Perform non-linear least-squares fitting on Optical Dipole Force (ODF) data.
"""

import json
import numpy as np
from pathlib import Path
from lmfit import Model
from typing import Tuple, Any


def sinc(x: np.ndarray, b: float, c: float, d: float) -> np.ndarray:
    """
    Calculates a sinc-squared function profile.

    This function follows the definition where sinc(x) = sin(pi*x)/(pi*x).

    Parameters
    ----------
    x : np.ndarray
        Independent variable (frequency or time).
    b : float
        Scaling factor for the frequency argument.
    c : float
        Amplitude of the sinc-squared peak.
    d : float
        Horizontal shift (centering) of the peak.

    Returns
    -------
    np.ndarray
        The computed sinc-squared values.
    """
    # np.sinc(y) in numpy is defined as sin(pi*y)/(pi*y)
    sinc_term = np.sinc(b * (x - d) / np.pi) ** 2  
    return c * sinc_term


def find_json_in_odf(filename: str) -> Path:
    """
    Searches parent directories for a folder named 'odf_json' and returns the file path.

    This utility climbs up the directory tree from the current file's location
    until it finds the specified data directory.

    Parameters
    ----------
    filename : str
        The name of the JSON file to search for.

    Returns
    -------
    Path
        The absolute path to the located file.

    Raises
    ------
    FileNotFoundError
        If the 'odf_json' folder or the specific file cannot be found.
    """
    current_path = Path(__file__).resolve()

    # Iterate through parent directories to find the data folder
    for parent in current_path.parents:
        odf_folder = parent / "odf_json"
        if odf_folder.exists() and odf_folder.is_dir():
            target_file = odf_folder / filename
            if target_file.exists():
                return target_file
            else:
                raise FileNotFoundError(
                    f"{filename} found in odf_json but the file itself does not exist"
                )

    raise FileNotFoundError("The 'odf_json' directory was not found in the path hierarchy")


def odf_data_fitting(
    max_frequency_mhz: float, 
    scan_points: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, np.ndarray]:
    """
    Loads experimental data from a JSON file and fits it to a sinc-squared model.

    Parameters
    ----------
    max_frequency_mhz : float
        The maximum frequency range for the generated frequency array.
    scan_points : int
        Number of points for the simulated frequency grid.

    Returns
    -------
    freq : np.ndarray
        Linearly spaced frequency array for plotting the fit.
    x_data : np.ndarray
        Experimental frequency data points.
    y_data : np.ndarray
        Experimental excitation values.
    result : lmfit.model.ModelResult
        The object containing fitting results and statistics.
    y_estimated : np.ndarray
        The fitted curve values computed over the 'freq' array.
    """
    # Create the frequency grid for estimation
    freq = np.linspace(-max_frequency_mhz, max_frequency_mhz, scan_points)

    # Locate the specific experimental dataset
    json_path = find_json_in_odf("18_01_35_174966.json")

    with open(json_path, "r") as f:
        dfile = json.load(f)

    # Extract experimental data: expected format is a 2D array
    data = np.array(dfile['data']['mean_excitation']['values'])

    # Initialize the lmfit Model based on the sinc function
    model = Model(sinc)

    # Set initial guesses for the parameters
    params = model.make_params(
        b=2263.33, 
        c=0.4067,  
        d=0.001
    )

    # data[:, 1] represents the x-axis (frequency) 
    # data[:, 0] represents the y-axis (excitation)
    x_data = data[:, 1]  
    y_data = data[:, 0]  

    # Execute the non-linear least-squares fit
    result = model.fit(y_data, params, x=x_data)

    # Generate the fitted curve using the optimized parameters
    y_estimated = sinc(freq, **result.best_values)

    return freq, x_data, y_data, result, y_estimated