import numpy as np







def outcome_simulator(self, current_measurement, noise_params=None, seed=None, laser_miscalibration=None, seed_miscalibration=None):

    # Step 2. At each measurement driven by the Bayesian Estimator, we calculate the outcome making the Simulator to evolve. 
    #       If that measurement is able to drive the transition and to effectively make the level to jump (according to the excitation probability distribution),
    #       then the outcome 1 is returned. Otherwise, the outcome is 0.

    # measurement values
    frequency = current_measurement[1]
    duration = current_measurement[2]
    dephasing = current_measurement[3]
    coherence_time = current_measurement[4]
    is_minus = current_measurement[5]
    rabi_rate_mhz = current_measurement[6]

    # Get the current state
    current_state = self.history_list[-1]
    current_index = current_state[0]

    # NEEDS TO BE MODIFIED. since Im considering only a single initial state, it is useless to compute the full matrix.
    # We only need to compute the column of probabilities that corresponds to the initial state ONLY.
    # Its tricky because when we will consider all the imperfections and jumps, the way the column will be computed will be different.
    # For the moment we keep this, that probably is not that efficient.
    # Probably instead of computing the likelihood now, one can precompute them a priori. However, if we are going to include time-dependent jumps,
    # this needs to be taken into account. But also in the Estimator. Probably a function for the jumps can be made, allowing to a priori computation of the likelihoods.
    
    exc_matrix = self.likelihoods_simulator(frequency=frequency,
                                            duration_us=duration,
                                            rabi_rate_mhz=rabi_rate_mhz,
                                            dephased=dephasing,
                                            coherence_time_us=coherence_time,
                                            is_minus=is_minus,
                                            noise_params=noise_params,
                                            seed=seed,
                                            maximum_excitation=0.9,
                                            laser_miscalibration=laser_miscalibration,
                                            seed_miscalibration=seed_miscalibration)

    # The excitation matrix is the sum of the lh0 and lh1.
    # I take the column correspondent to the initial state, where I have all the probabilities to be excited and the probability to stay in the same initial state.
    column = exc_matrix[:, current_index]

    # I randomly sample the final state according to the excitation probability distribution. It can be the same initial state or the driven new state 
    new_state_index = self.new_state_index(column)

    # From the index of the final state, I retrieve the information of it: J, m, xi
    selected_row = self.model.state_df.loc[new_state_index]
    j_val = selected_row["j"]
    m_val = selected_row["m"]
    xi_val = selected_row["xi"]


    # The final state is appended to the history list.
    # The outcome of the transition is then returned.
    new_state = [new_state_index, j_val, m_val, xi_val]
    self.history_list.append(new_state)


    if current_index == new_state_index:
        outcome = 0

        # False positive rate
        if np.random.rand() < self.fpr:
            outcome = 1
    else:
        outcome = 1

        # False negative rate
        if np.random.rand() < self.fnr:
            outcome = 0

    return outcome
