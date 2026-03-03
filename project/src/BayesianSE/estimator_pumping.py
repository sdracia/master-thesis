import numpy as np



# optical pumping: restores the prior
# The pumping here is done with the pulse at -350 Hz at Dm = -1 and the one at +150 Hz at Dm = -1.
def reset_prior_pumping(self, simulator, save_data):
    print("repump")
    # Restore the state distribution to the one after the total pumping
    # It cancels any prior believe of the state distribution, as if we did no measurements yet
    pop_fit = self.pump_efficiencies
    pop_fit_1 = pop_fit[1]  # pop_fit_1
    pop_fit_2jp1 = pop_fit[2]  # pop_fit_2jp1

    df = self.model.state_df.copy()

    # ESTIMATOR
    self.prior = self.after_pumping_dist

    if save_data:   # Need this because length of history_list["posterior"] has to be the same of history_list of the simulator
        self.history_list.append({
            "posterior": self.prior.tolist()
        })

    # SIMULATOR
    current_state = simulator.history_list[-1]
    current_index = current_state[0]
    current_j = current_state[1]

    indices_in_j = df.index[df["j"] == current_j]
    m_len = 2* current_j + 1

    if current_index in indices_in_j[1:m_len]:
        # p1 is probabiity of being in pen upper; p0 = probabiity of being in leftmost
        # I extract a random number and the new state end up in one of the 2 possibilties in the same manifold
        p1 = min(2 * pop_fit_1[current_j-1], 1.0)
        p0 = 1 - p1

        new_state_index = np.random.choice(
            [indices_in_j[1], indices_in_j[0]],
            p=[p1, p0]
        )

    if current_index in indices_in_j[m_len:]:
        # p1 is probabiity of being in pen lower; p0 = probabiity of being in leftmost
        # I extract a random number and the new state end up in one of the 2 possibilties in the same manifold
        p1 = min(2 * pop_fit_2jp1[current_j-1], 1.0)
        p0 = 1 - p1

        new_state_index = np.random.choice(
            [indices_in_j[m_len], indices_in_j[0]],
            p=[p1, p0]
        )

    else:
        # If the current state is the leftmost state, I just keep it
        new_state_index = indices_in_j[0]

    selected_row = df.loc[new_state_index]
    j_val = selected_row["j"]
    m_val = selected_row["m"]
    xi_val = selected_row["xi"]
        
    new_state = [new_state_index, j_val, m_val, xi_val]
    simulator.history_list.append(new_state)

# Function for repumping without restoring the prior
def within_run_pumping(self, simulator, save_data):
    ## Can be used:
    #  - If the variance of the estimator is too big, we can pump the believe with this function
    #  - When we apply measurements with high rabi rates that excite multiple manifolds: off-resonant coupling
    #       will happen, and so this function could repump the population to the leftmost/penultimate state.
        
    pop_fit = self.pump_efficiencies
    pop_fit_1 = pop_fit[1]  # pop_fit_1
    pop_fit_2jp1 = pop_fit[2]  # pop_fit_2jp1

    df = self.model.state_df.copy()

    # ESTIMATOR
    distribution = self.prior.copy()

    initial_total = distribution.sum()


    for j_val in range(1, self.j_max + 1):
        
        m_len = 2* j_val + 1
        indices_in_j = df.index[df["j"] == j_val]

        if len(indices_in_j) != 2*m_len:
            raise ValueError(f"Error occurred in filtering the dataframe")


        before_sum = distribution[indices_in_j].sum()

        leftmost_index = indices_in_j[0]
        pen_upper_index = indices_in_j[1]
        pen_lower_index = indices_in_j[m_len]

        upper_manif_indices = indices_in_j[2:m_len]
        low_manif_indices = indices_in_j[m_len+1:]
        
        sum_upper = distribution[pen_upper_index] + distribution[upper_manif_indices].sum()

        pop_on_1 = sum_upper*0.95
        distribution[pen_upper_index] = pop_on_1
        distribution[upper_manif_indices] = sum_upper*0.05/(m_len-2)

        distribution[pen_upper_index] = pop_fit_1[j_val-1]*2 * pop_on_1
        distribution[leftmost_index] += (1 - pop_fit_1[j_val-1]*2) * pop_on_1


        sum_lower = distribution[pen_lower_index] + distribution[low_manif_indices].sum()

        pop_on_2jp1 = sum_lower*0.95
        distribution[pen_lower_index] = pop_on_2jp1
        distribution[low_manif_indices] = sum_lower*0.05/(m_len-1)

        distribution[pen_lower_index] = pop_fit_2jp1[j_val-1]*2 * pop_on_2jp1
        distribution[leftmost_index] += (1 - pop_fit_2jp1[j_val-1]*2) * pop_on_2jp1

        after_sum = distribution[indices_in_j].sum()
        if not np.isclose(before_sum, after_sum, atol=1e-8):
            raise ValueError(f"Probability distribution inside the manifold is not conserved")
    
    # Checks
    final_total = distribution.sum()
    if not np.isclose(initial_total, final_total, atol=1e-7):
        raise ValueError(f"Overall probability distribution is not conserved")

    if not np.isclose(final_total, 1, atol=1e-8):
        raise ValueError(f"Probability distribution does not sum up to 1")

    if (distribution < -1e-10).any():
        raise ValueError(f"Negative values found in distribution: min = {distribution.min()}")

    self.prior = distribution
    
    if save_data:   # Need this because length of history_list["posterior"] has to be the same of history_list of the simulator
        self.history_list.append({
            "posterior": self.prior.tolist()
        })

    
    # SIMULATOR
    current_state = simulator.history_list[-1]
    current_index = current_state[0]
    current_j = current_state[1]

    indices_in_j = df.index[df["j"] == current_j]
    m_len = 2* current_j + 1

    if current_index in indices_in_j[1:m_len]:
        # p1 is probabiity of being in pen upper; p0 = probabiity of being in leftmost
        # I extract a random number and the new state end up in one of the 2 possibilties in the same manifold
        p1 = min(2 * pop_fit_1[current_j-1], 1.0)
        p0 = 1 - p1

        new_state_index = np.random.choice(
            [indices_in_j[1], indices_in_j[0]],
            p=[p1, p0]
        )

    if current_index in indices_in_j[m_len:]:
        # p1 is probabiity of being in pen lower; p0 = probabiity of being in leftmost
        # I extract a random number and the new state end up in one of the 2 possibilties in the same manifold
        p1 = min(2 * pop_fit_2jp1[current_j-1], 1.0)
        p0 = 1 - p1

        new_state_index = np.random.choice(
            [indices_in_j[m_len], indices_in_j[0]],
            p=[p1, p0]
        )

    else:
        # If the current state is the leftmost state, I just keep it
        new_state_index = indices_in_j[0]

    selected_row = df.loc[new_state_index]
    j_val = selected_row["j"]
    m_val = selected_row["m"]
    xi_val = selected_row["xi"]
        
    new_state = [new_state_index, j_val, m_val, xi_val]
    simulator.history_list.append(new_state)



