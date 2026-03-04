import numpy as np
import random


from BayesianSE.estimator import BayesianStateEstimation
from BayesianSE.simulator import BayesianSimulator
from BayesianSE._plotting import plot_cross_entropies_vs_step, histo_final_cross_entropy
from BayesianSE._statistics import compute_variance, compute_statistics
from BayesianSE._utils import cleaning_convergence
from BayesianSE._noise_models import var_misfreq
from BayesianSE._run_manager import init_new_run

from BayesianSE._run_manager import save_metadata, save_results



def run_bayesian_state_estimation(molecule=None, molecule_type="CaOH", temperature=300, b_field_gauss=3.27, j_max=50,
                                  rabi_by_j=2*np.pi*0.001, dephased=False, coherence_time_us=1000.0, is_minus=False,
                                  false_positive_rate=0.0, false_negative_rate=0.0,
                                  noise_params=None, seed=None, laser_miscalibration=None, seed_miscalibration=None,
                                  noise_params_estim=None, laser_miscalibration_estim=None,
                                  pop_fit=None,
                                  N=100, num_updates=10, block_steps=1, type_block=None, 
                                  apply_pumping=False, marginalization=True, false_rates=True,
                                  save_data=True, only_total=False,
                                  max_excitation=0.9
                                  ):

    curves_by_label = {
        'LM': [],
        'PU': [],
        'PL': [],
        'Other': []
    }

    j_samples = []
    xi_samples = []
    m_samples = []


    color_map = {
        'LM': '#800020',
        'PU': '#4B0082',
        'PL': '#DAA520',
        'Other': '#0C630B'
    }




    Estimator = BayesianStateEstimation(model=molecule, temperature=temperature, b_field_gauss=b_field_gauss, j_max=j_max,
                                        false_positive_rate=false_positive_rate, false_negative_rate=false_negative_rate)

    Estimator.measurement_setting(
        rabi_by_j=rabi_by_j, 
        dephased=dephased, 
        coherence_time_us=coherence_time_us, 
        is_minus=is_minus, 
        noise_params=noise_params_estim, 
        seed=None, 
        max_excitation=max_excitation,
        laser_miscalibration=laser_miscalibration_estim,
        seed_miscalibration=None,
        marginalization = marginalization
    )


    variance_by_label = {
        'LM': [],
        'PU': [],
        'PL': [],
        'Other': []
    }


    misfrequency_by_label = {
        'LM': [],
        'PU': [],
        'PL': [],
        'Other': []
    }



    for run in range(N):
        seed_miscalibration = random.randint(0, 10000)

        print("Run = ", run)
        

        Simulator = BayesianSimulator(model=molecule, temperature=temperature, b_field_gauss=b_field_gauss, j_max=j_max, 
                                    false_positive_rate=false_positive_rate, false_negative_rate=false_negative_rate)  

        j_sample = Simulator.history_list[0][1]
        xi_sample = Simulator.history_list[0][3]
        m_sample = Simulator.history_list[0][2]

        j_samples.append(j_sample)  
        xi_samples.append(xi_sample)
        m_samples.append(m_sample)

        
        Estimator.update_distibution(Simulator, 
                                    num_updates=num_updates, 
                                    apply_pumping=apply_pumping, 
                                    save_data=save_data, 
                                    block_steps=block_steps, 
                                    type_block=type_block,
                                    noise_params=noise_params,
                                    seed=None,
                                    laser_miscalibration=laser_miscalibration,
                                    seed_miscalibration=seed_miscalibration,
                                    pop_fit=pop_fit,
                                    false_rates=false_rates,
                                    max_excitation=max_excitation
                                    )
        
        cross_entropy = Estimator.cross_entropy(Simulator)

        sim_init_state = Simulator.history_list[0]
        sim_init_j_val = sim_init_state[1]
        sim_init_m_val = sim_init_state[2]
        sim_init_xi_val = sim_init_state[3]

        if sim_init_m_val == -sim_init_j_val - 0.5:
            label = 'LM'
        elif sim_init_m_val == -sim_init_j_val + 0.5 and not bool(sim_init_xi_val):
            label = 'PU'
        elif sim_init_m_val == -sim_init_j_val + 0.5 and bool(sim_init_xi_val):
            label = 'PL'
        else:
            label = 'Other'

        steps = list(range(len(cross_entropy)))
        curves_by_label[label].append((steps, cross_entropy))

        freq = Simulator.misfrequency
        misfrequency_by_label[label].append((freq, cross_entropy[-1]))

        last_post = np.array(Estimator.history_list[-1]["posterior"])
        last_variance = compute_variance(last_post)

        variance_by_label[label].append((last_variance, cross_entropy[-1]))



        Estimator.init_prior()
        Estimator.history_list = []



    init_new_run()

    plot_cross_entropies_vs_step(curves_by_label, color_map, filename="run_CE.svg") ###


    total_runs, success_rate_total, leftmost_total, success_rate_leftmost = histo_final_cross_entropy(curves_by_label, color_map, only_total, filename="run_CEhist.svg") ###


    stats, all_step, below_step, above_step = compute_statistics(curves_by_label, num_updates, block_steps, filename="run_stats.svg") ###


    var_misfreq(variance_by_label, misfrequency_by_label, filename="run_varmisfreq.svg")   ###


    max_step = num_updates * block_steps

    (curves_by_label_filtered, misfrequency_by_label_filtered, variance_by_label_filtered,
    curves_by_label_not_converged, misfrequency_by_label_not_converged, variance_by_label_not_converged,
    fraction_converged, fraction_not_converged) = cleaning_convergence(
        max_step, curves_by_label, misfrequency_by_label, variance_by_label
    )

    print(f"Converging runs fraction: {fraction_converged:.2%}")
    print(f"NON-Converging runs fraction: {fraction_not_converged:.2%}")

    print("###########################")
    print("##### Converging runs #####")
    print("############################")
    total_filtered, success_rate_filtered, _, _ = histo_final_cross_entropy(curves_by_label_filtered, color_map, only_total, filename="run_CEhist_conv.svg")

    stats_filtered, all_step_filtered, below_step_filtered, above_step_filtered = compute_statistics(curves_by_label_filtered, num_updates, block_steps, plot = False, filename="run_stats_conv.svg")

    var_misfreq(variance_by_label_filtered, misfrequency_by_label_filtered)

    print("###########################")
    print("### NON-Converging runs ###")
    print("############################")
    total_not_conv, success_rate_not_conv, _, _ = histo_final_cross_entropy(curves_by_label_not_converged, color_map, only_total, filename="run_CEhist_nonconv.svg")

    stats_not_conv, all_step_not_conv, below_step_not_conv, above_step_not_conv = compute_statistics(curves_by_label_not_converged, num_updates, block_steps, plot = False, filename="run_stats_nonconv.svg")

    var_misfreq(variance_by_label_not_converged, misfrequency_by_label_not_converged)

    results = {
        "total_runs": total_runs,
        "success_rate_total": success_rate_total,
        "leftmost_total": leftmost_total,
        "success_rate_leftmost": success_rate_leftmost,
        "stats": stats,
        "all_step": all_step,
        "below_step": below_step,
        "above_step": above_step,
        
        "total_filtered": total_filtered,
        "success_rate_filtered": success_rate_filtered,
        "stats_filtered": stats_filtered,
        "all_step_filtered": all_step_filtered,
        "below_step_filtered": below_step_filtered,
        "above_step_filtered": above_step_filtered,
        
        "total_not_conv": total_not_conv,
        "success_rate_not_conv": success_rate_not_conv,
        "stats_not_conv": stats_not_conv,
        "all_step_not_conv": all_step_not_conv,
        "below_step_not_conv": below_step_not_conv,
        "above_step_not_conv": above_step_not_conv,
        
        "fraction_converged": fraction_converged,
        "fraction_not_converged": fraction_not_converged
    }


    save_metadata(molecule_type, temperature, b_field_gauss, j_max, rabi_by_j, dephased, coherence_time_us, is_minus,
          false_positive_rate, false_negative_rate, noise_params, seed, laser_miscalibration, seed_miscalibration,
          noise_params_estim, laser_miscalibration_estim, pop_fit, N, num_updates,
          block_steps, type_block, apply_pumping, marginalization, false_rates, save_data, only_total, max_excitation)
    
    save_results(results)
    
    return results


        



    