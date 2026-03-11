# File: simulator_utils.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Module for Analyzing Molecular Rotational Distributions.

This module provides utilities to extract and aggregate population data 
from molecular state dataframes, specifically focusing on calculating 
the total population for each rotational quantum number (J) manifold.
"""

import numpy as np
from molecules.molecule import Molecule


def j_distribution(
    self, 
    molecule: Molecule, 
    j_max: int
) -> np.ndarray:
    """
    Calculates the total population for each rotational manifold J.

    This function iterates through the rotational manifolds from J=0 up to 
    the specified j_max, identifies all internal states belonging to each 
    manifold, and sums their respective populations.

    Parameters
    ----------
    molecule : Molecule
        The molecular object containing the state dataframe with 
        population data.
    j_max : int
        The maximum rotational quantum number J to include in the 
        distribution analysis.

    Returns
    -------
    np.ndarray
        A NumPy array where the i-th element represents the total 
        population of the manifold with J = i.
    """
    list_pop = []

    for j_val in range(0, j_max + 1):
        
        indices_in_j = molecule.state_df.index[
            molecule.state_df["j"] == j_val
        ]
        
        total_in_j = molecule.state_df.loc[indices_in_j, "state_dist"].sum()

        list_pop.append(total_in_j)

    return np.array(list_pop)