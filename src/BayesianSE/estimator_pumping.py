# File: estimator_pumping.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Pumping and Repumping Module for Bayesian State Estimation.

This module provides functions to handle the 'repumping' of molecular states 
during a Bayesian estimation sequence. It includes logic to:
- Reset the estimator's prior belief to the initial post-pumping distribution.
- Physically update the simulator's state based on pumping efficiencies.
- Perform intermediate 'within-run' pumping to redistribute populations 
  without fully resetting the estimator's memory.
"""
from typing import Any
import numpy as np


def reset_prior_pumping(self, simulator: Any, save_data: bool) -> None:
    """
    Resets the estimator's prior to the initial post-pumping state.

    This function restores the belief of the state distribution to the one 
    calculated immediately after the initial pumping stage. It effectively 
    erases the impact of any measurements performed so far in the current run.

    Parameters
    ----------
    simulator : object
        The ground-truth molecular simulator object.
    save_data : bool
        If True, appends the restored prior to the estimator's history list.
    """
    print("repump")

    # Access pre-calculated pumping efficiencies for sigmoid fits
    pop_fit = self.pump_efficiencies
    pop_fit_1 = pop_fit[1]      # Population efficiency for penultimate upper state
    pop_fit_2jp1 = pop_fit[2]   # Population efficiency for penultimate lower state

    df = self.model.state_df.copy()

    # --- ESTIMATOR LOGIC ---
    # Restore the belief distribution (prior) to the post-pumping baseline
    self.prior = self.after_pumping_dist

    if save_data:
        # history_list length must stay synchronized with the physical simulator
        self.history_list.append({
            "posterior": self.prior.tolist()
        })

    # --- SIMULATOR LOGIC (Physical state update) ---
    current_state = simulator.history_list[-1]
    current_index = current_state[0]
    current_j = current_state[1]

    indices_in_j = df.index[df["j"] == current_j]
    m_len = 2 * current_j + 1

    # Logic to redistribute the physical state based on current location in the J-manifold
    if current_index in indices_in_j[1:m_len]:
        # Calculate transition probability to penultimate upper or leftmost state
        p1 = min(2 * pop_fit_1[current_j - 1], 1.0)
        p0 = 1 - p1

        new_state_index = np.random.choice(
            [indices_in_j[1], indices_in_j[0]],
            p=[p1, p0]
        )

    if current_index in indices_in_j[m_len:]:
        # Calculate transition probability to penultimate lower or leftmost state
        p1 = min(2 * pop_fit_2jp1[current_j - 1], 1.0)
        p0 = 1 - p1

        new_state_index = np.random.choice(
            [indices_in_j[m_len], indices_in_j[0]],
            p=[p1, p0]
        )
    else:
        # If the molecule is in the leftmost state, it remains there
        new_state_index = indices_in_j[0]

    # Update the physical simulator's history with the new state parameters
    selected_row = df.loc[new_state_index]
    j_val = selected_row["j"]
    m_val = selected_row["m"]
    xi_val = selected_row["xi"]
        
    new_state = [new_state_index, j_val, m_val, xi_val]
    simulator.history_list.append(new_state)


def within_run_pumping(self, simulator: Any, save_data: bool) -> None:
    """
    Applies a repumping stage to the current belief without resetting it.

    This function is used to sharpen the estimator's belief or to account for 
    off-resonant coupling during measurements. It redistributes the probability 
    mass within each manifold towards the target states based on the pumping 
    model.

    Parameters
    ----------
    simulator : object
        The ground-truth molecular simulator object.
    save_data : bool
        If True, appends the updated prior to the estimator's history list.

    Raises
    ------
    ValueError
        If the probability distribution is not conserved or contains negative values.
    """
    pop_fit = self.pump_efficiencies
    pop_fit_1 = pop_fit[1]
    pop_fit_2jp1 = pop_fit[2]

    df = self.model.state_df.copy()

    # --- ESTIMATOR LOGIC ---
    distribution = self.prior.copy()
    initial_total = distribution.sum()

    for j_val in range(1, self.j_max + 1):
        m_len = 2 * j_val + 1
        indices_in_j = df.index[df["j"] == j_val]

        if len(indices_in_j) != 2 * m_len:
            raise ValueError(f"Error occurred in filtering the dataframe")

        before_sum = distribution[indices_in_j].sum()

        # Specific state indices within the current J manifold
        leftmost_index = indices_in_j[0]
        pen_upper_index = indices_in_j[1]
        pen_lower_index = indices_in_j[m_len]

        upper_manif_indices = indices_in_j[2:m_len]
        low_manif_indices = indices_in_j[m_len + 1:]
        
        # --- Upper Manifold Redistribution ---
        sum_upper = distribution[pen_upper_index] + distribution[upper_manif_indices].sum()

        pop_on_1 = sum_upper * 0.95
        distribution[pen_upper_index] = pop_on_1
        distribution[upper_manif_indices] = sum_upper * 0.05 / (m_len - 2)

        # Final belief update for upper state based on pumping fit
        distribution[pen_upper_index] = pop_fit_1[j_val - 1] * 2 * pop_on_1
        distribution[leftmost_index] += (1 - pop_fit_1[j_val - 1] * 2) * pop_on_1

        # --- Lower Manifold Redistribution ---
        sum_lower = distribution[pen_lower_index] + distribution[low_manif_indices].sum()

        pop_on_2jp1 = sum_lower * 0.95
        distribution[pen_lower_index] = pop_on_2jp1
        distribution[low_manif_indices] = sum_lower * 0.05 / (m_len - 1)

        # Final belief update for lower state based on pumping fit
        distribution[pen_lower_index] = pop_fit_2jp1[j_val - 1] * 2 * pop_on_2jp1
        distribution[leftmost_index] += (1 - pop_fit_2jp1[j_val - 1] * 2) * pop_on_2jp1

        # Conservation check within the manifold
        after_sum = distribution[indices_in_j].sum()
        if not np.isclose(before_sum, after_sum, atol=1e-8):
            raise ValueError("Probability distribution inside the manifold is not conserved")
    
    # --- Final Consistency Checks ---
    final_total = distribution.sum()
    if not np.isclose(initial_total, final_total, atol=1e-7):
        raise ValueError("Overall probability distribution is not conserved")

    if not np.isclose(final_total, 1, atol=1e-8):
        raise ValueError("Probability distribution does not sum up to 1")

    if (distribution < -1e-10).any():
        raise ValueError(f"Negative values found in distribution: min = {distribution.min()}")

    # Update the estimator's internal prior
    self.prior = distribution
    
    if save_data:
        self.history_list.append({
            "posterior": self.prior.tolist()
        })

    # --- SIMULATOR LOGIC (Physical state update) ---
    current_state = simulator.history_list[-1]
    current_index = current_state[0]
    current_j = current_state[1]

    indices_in_j = df.index[df["j"] == current_j]
    m_len = 2 * current_j + 1

    if current_index in indices_in_j[1:m_len]:
        p1 = min(2 * pop_fit_1[current_j - 1], 1.0)
        p0 = 1 - p1
        new_state_index = np.random.choice(
            [indices_in_j[1], indices_in_j[0]],
            p=[p1, p0]
        )

    if current_index in indices_in_j[m_len:]:
        p1 = min(2 * pop_fit_2jp1[current_j - 1], 1.0)
        p0 = 1 - p1
        new_state_index = np.random.choice(
            [indices_in_j[m_len], indices_in_j[0]],
            p=[p1, p0]
        )
    else:
        new_state_index = indices_in_j[0]

    selected_row = df.loc[new_state_index]
    j_val = selected_row["j"]
    m_val = selected_row["m"]
    xi_val = selected_row["xi"]
        
    new_state = [new_state_index, j_val, m_val, xi_val]
    simulator.history_list.append(new_state)