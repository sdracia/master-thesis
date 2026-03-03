import numpy as np


from molecules.molecule import CaH, CaOH
from ._statistics import compute_variance




def update_distibution(
        self, 
        simulator, 
        num_updates, 
        apply_pumping = False, 
        save_data = True, 
        block_steps = 1, 
        type_block = None, 
        noise_params=None, 
        seed=None, 
        laser_miscalibration=None, 
        seed_miscalibration=None,
        pop_fit = None,
        false_rates = True
        ):
    
    if isinstance(self.model, CaH):
        threshold_variance = 2500
    elif isinstance(self.model, CaOH):
        threshold_variance = 0.15*1e6

    if apply_pumping:
        if pop_fit is not None: 
            self.pump_efficiencies = pop_fit
        else:
            raise ValueError("Need efficiencies for pumping")


    final_step = num_updates * block_steps
    stop_flag = False
    variance = 0.0
    n_wrong_sweep = 0.0

    for j in range (num_updates):

        # After the first sweep, if at the beginning of the sweep the prior has a variance smaller than a certain
        # threshold then we repump the Simulator and at the same time we repump the probability distribution
        # of the posterior on the Estimator part. It is an update of the posterior without detecting an outcome.
        
        if apply_pumping:
            if j != 0 and j%self.j_max == 0 and variance > threshold_variance:
                print("Re-pumping updating prior")
                self.within_run_pumping(simulator, save_data)
                
        # Part to change the frequency of the measurements after each failure sweep.
        if j != 0 and j%(self.j_max) == 0 and j//self.j_max >= 6:
            # Part of the first strategy for block2 and block3 only
            n_wrong_sweep += 1



        for i in range(block_steps):
            
            self.meas_idx = self.get_next_setting(j, i, ty = type_block)

            data = self.prior

            lh0 = self.Probs_exc_list[self.meas_idx][0]
            lh1 = self.Probs_exc_list[self.meas_idx][1]

            current_measurement = self.measurements[self.meas_idx]

            self.outcome = simulator.outcome_simulator(current_measurement, noise_params, seed, laser_miscalibration, seed_miscalibration)

            if false_rates:
                if self.outcome == 0:
                    # likelihood = lh0
                    likelihood = (1 - self.fnr) * lh0 + self.fpr * lh1
                else:
                    # likelihood = lh1
                    likelihood = self.fnr * lh0 + (1 - self.fpr) * lh1
            else:
                if self.outcome == 0:
                    likelihood = lh0
                else:
                    likelihood = lh1
            
            posterior = likelihood.dot(data)
            posterior = posterior / np.sum(posterior)


            self.likelihood = likelihood
            self.posterior = posterior

            if save_data:
                self.history_list.append({
                    "meas_idx": self.meas_idx,
                    "measurement": self.measurements[self.meas_idx],
                    "outcome": self.outcome,
                    "prior": self.prior.tolist(),
                    "likelihood": self.likelihood,
                    "posterior": self.posterior.tolist()
                })

            # prior updating:
            self.prior = posterior

            variance = compute_variance(self.prior)

            if self.stop_condition(self.prior):
                final_step = (j+1)*block_steps
                print(f"Stop condition reached at step {final_step}")

                stop_flag = True
                break

        if stop_flag:
            break




def get_next_setting(self, j, i, ty = None):

    j_max = self.j_max

    direct_meas = j_max - 1 - (j % j_max)
    inverse_meas = j_max - 1 - (j % j_max) + j_max

    if ty is None:
        idx = direct_meas

    # # Block 1
    if ty == "block1":  # u-d-d
        if i == 0:
            idx = direct_meas
        else:
            idx = inverse_meas

    # # Block 2
    if ty == "block2":  # u-d-u
        if i == 0 or i == 2:    
            idx = direct_meas
        else:
            idx = inverse_meas

    # # Block 3
    if ty == "block3":  # u-d
        if i == 0:
            idx = direct_meas
        else:
            idx = inverse_meas

    # # Block 4
    if ty == "block4":  # u-d-u-d
        if i == 0 or i == 2:
            idx = direct_meas
        else:
            idx = inverse_meas

    # Block 5
    if ty == "block5":  # u-d-d-u-d-d
        if i == 0 or i == 3:
            idx = direct_meas
        else:
            idx = inverse_meas

    if ty == "block6":  # u-u-u-d-d-d-u-u-u
        if i == 0 or i == 1 or i == 2 or i == 6 or i == 7 or i == 8:   
            idx = direct_meas
        else:
            idx = inverse_meas

    if ty == "block7":  # u-u-u-u-d-d-d-d-u-u-u-u
        if i == 0 or i == 1 or i == 2 or i == 3 or i == 8 or i == 9 or i == 10 or i == 11:
            idx = direct_meas
        else:
            idx = inverse_meas

    if ty == "block8":  # u-d-u-d-u-d-u-d-u-d-u-d-u-d-u-d-u-d-u-d-u-d-u-d
        if i == 0 or i == 2 or i == 4 or i == 6 or i == 8 or i == 10 or i == 12 or i == 14 or i == 16 or i == 18 or i == 20 or i == 22:
            idx = direct_meas
        else:
            idx = inverse_meas

    if ty == "block9":  # u-u-u-u-d-d-d-d-u-u-u-u-d-d-d-d-u-u-u-u-d-d-d-d
        if i == 0 or i == 1 or i == 2 or i == 3 or i == 8 or i == 9 or i == 10 or i == 11 or i==16 or i == 17 or i == 18 or i == 19:
            idx = direct_meas
        else:
            idx = inverse_meas


    return idx
