import numpy as np
import matplotlib.pyplot as plt


from ._noise_models import build_detuning_distributions
from saving import save_figure_in_images


def measurement_setting(
    self, 
    rabi_by_j, 
    dephased, 
    coherence_time_us, 
    is_minus, 
    noise_params = None, 
    seed = None, 
    max_excitation = 0.9,
    laser_miscalibration = None,
    seed_miscalibration = None,
    marginalization = True):

    # Avoid marginalization when the sum of the "frequency" on laser_miscalibration and noise_params 
    # is 0, even when None dictionaries are passed or when "frequency" is not a key.      

    freq_mis_level = (laser_miscalibration or {}).get("frequency", {}).get("level", 0.0)
    freq_noise_level = (noise_params or {}).get("frequency", {}).get("level", 0.0)

    if freq_mis_level + freq_noise_level == 0.0:
        marginalization = False


    # the laser field fixes the rabi_rate: it's the same for the measurements that drive transitions and optical pumping.
    # [dephased, coh_time, is_minus] are considered the same for all measurements. can be done differently
    df_trans = self.model.transition_df

    if isinstance(rabi_by_j, dict):
        measurements = []

        for (j_min, j_max), rabi_rate_mhz in rabi_by_j.items():
            for j in range(j_min, j_max + 1):

                # These should be computed considering my believe of the molecular structure, so not considering the molecular fluctuations
                measurements.append([
                    j,
                    (-1 if not is_minus else 1) * df_trans.loc[df_trans["j"] == j].iloc[0]["energy_diff"] * 1e-3, 
                    np.pi/(rabi_rate_mhz*df_trans.loc[df_trans["j"]==j].iloc[0]["coupling"]), 
                    dephased, 
                    coherence_time_us,
                    is_minus,
                    rabi_rate_mhz
                ])
        
        is_minus = not is_minus

        for (j_min, j_max), rabi_rate_mhz in rabi_by_j.items():
            for j in range(j_min, j_max + 1):

                measurements.append([
                    j,
                    (-1 if not is_minus else 1) * df_trans.loc[df_trans["j"] == j].iloc[0]["energy_diff"] * 1e-3,
                    np.pi/(rabi_rate_mhz*df_trans.loc[df_trans["j"]==j].iloc[0]["coupling"]),
                    dephased,
                    coherence_time_us,
                    is_minus,
                    rabi_rate_mhz
                ])

    elif isinstance(rabi_by_j, (int, float)):
        measurements = [[j,
                    (-1 if not is_minus else 1) * df_trans.loc[df_trans["j"] == j].iloc[0]["energy_diff"] * 1e-3, 
                    np.pi/(rabi_by_j*df_trans.loc[df_trans["j"]==j].iloc[0]["coupling"]), 
                    dephased, 
                    coherence_time_us,
                    is_minus,
                    rabi_by_j
                    ] for j in range(1, self.j_max+1)]

        is_minus = not is_minus
        
        measurements += [[j,
                (-1 if not is_minus else 1) * df_trans.loc[df_trans["j"] == j].iloc[0]["energy_diff"] * 1e-3,
                np.pi/(rabi_by_j*df_trans.loc[df_trans["j"]==j].iloc[0]["coupling"]),
                dephased,
                coherence_time_us,
                is_minus,
                rabi_by_j
                ] for j in range(1, self.j_max+1)]
    else:
        raise TypeError("rabi_rate_mhz must be either a single float or a dictionary mapping ranges to values.")
    
    self.Probs_exc_list = []

    if marginalization == True:
        for j,  frequency, duration, deph, coh_time, is_min, rabi in measurements:

            freq_grid, freq_probs = build_detuning_distributions(
                frequency=frequency,
                rabi_rate=rabi,
                laser_miscalibration=laser_miscalibration,
                noise_params=noise_params
            )

            lh0, lh1 = 0.0, 0.0

            for f, p in zip(freq_grid, freq_probs):

                l0, l1 = self.likelihoods_estimator(
                    frequency=f,
                    duration_us=duration,
                    rabi_rate_mhz=rabi,
                    dephased=deph,
                    coherence_time_us=coh_time,
                    is_minus=is_min,
                    noise_params=None,  # already considered in the marginalization
                    seed=None,
                    maximum_excitation=max_excitation,
                    laser_miscalibration=None,  # already considered in the marginalization
                    seed_miscalibration=None
                )

                lh0 += p * l0
                lh1 += p * l1

            # CHECK FOR NEGATIVE VALUES AND NORMALIZATIONS
            exc_mat = lh0+lh1

            tol_neg = 1e-6  # tolerance for the negligible negative values 
            negative_mask = exc_mat.data < - tol_neg
            if np.any(negative_mask):
                raise ValueError(f"Significant negative values found in the matrix: {exc_mat.data[negative_mask]}")
            
            row_sums = np.array(exc_mat.sum(axis=0)).flatten()
            tol_row = 1e-3  # tolerance for the sum in the rows
            bad_rows = np.abs(row_sums - 1.0) > tol_row
            if np.any(bad_rows):
                raise ValueError(f"The sum of some rows is signicantly different from 1: {row_sums[bad_rows]}")


            # O outcome = transition not happened
            # 1 outcome = transition happened
            self.Probs_exc_list.append((lh0, lh1))

    elif marginalization == False: 

        for _,  frequency, duration, deph, coh_time, is_min, rabi in measurements:

            lh0, lh1 = self.likelihoods_estimator(
                frequency=frequency,
                duration_us=duration,
                rabi_rate_mhz=rabi,
                dephased=deph,
                coherence_time_us=coh_time,
                is_minus=is_min,
                noise_params=noise_params,
                seed=seed,
                maximum_excitation=max_excitation,
                laser_miscalibration=laser_miscalibration,
                seed_miscalibration=seed_miscalibration
            )

            # O outcome = transition not happened
            # 1 outcome = transition happened
            self.Probs_exc_list.append((lh0, lh1))

    else:
        raise ValueError(f"Marginalization variable should be either True or False")


    self.measurements = measurements
    



def meas_sensitivity_heatmap(Estimator, final_index, initial_index, figname = "meas_sensitivity.svg", title=None):
    measurements = Estimator.measurements
    probabilities = Estimator.Probs_exc_list
    j_max = Estimator.j_max

    heatmap_data = np.zeros((j_max, j_max))

    for meas_idx in range(j_max):
        _, lh_matrix = probabilities[meas_idx]
        for j in range(1, j_max+1):
            # Estrai il valore dalla matrice sparsa
            value = lh_matrix[final_index[j], initial_index[j]]
            heatmap_data[meas_idx, j-1] = value  # j-1 per indice 0-based


    fig, ax = plt.subplots(figsize=(7, 5))

    # Mostra la heatmap e salva l'oggetto immagine per la colorbar
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')

    # Aggiungi colorbar con controllo sull'asse
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Excitation probability", fontsize=25)
    cbar.ax.tick_params(labelsize=18)

    # Etichette e titolo
    ax.set_xlabel(r"$J$ signature transition", fontsize=24)
    ax.set_ylabel(r"J measurement pulse $\mu_J^{+}$", fontsize=24)
    if title is None:
        title = "Measurement sensitivity"

    ax.set_title(title, fontsize=28)
    # Tick ogni 5
    ax.set_xticks(np.arange(0, j_max, 5))
    ax.set_xticklabels(np.arange(1, j_max + 1, 5))
    ax.set_yticks(np.arange(0, j_max, 5))
    ax.set_yticklabels(np.arange(1, j_max + 1, 5))

    ax.tick_params(axis='both', pad=10, labelsize=20)

    plt.tight_layout()

    save_figure_in_images(fig, filename=figname)

    plt.show()
