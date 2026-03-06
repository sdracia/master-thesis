"""
Main Orchestration Module for Bayesian State Estimation (BSE) Experiments.

This module provides the high-level pipeline to execute Monte Carlo simulations 
of molecular state estimation. It integrates the physical simulator and 
the Bayesian estimator to:
- Evaluate the convergence of the estimation across different rotational manifolds.
- Categorize performance based on initial molecular states (LM, PU, PL).
- Handle experimental noise, miscalibration, and marginalization.
- Generate a comprehensive suite of statistical plots and save results/metadata.
"""

import numpy as np
import random
from typing import Optional, Dict, Any, Union, List

from BayesianSE.estimator import BayesianStateEstimation
from BayesianSE.simulator import BayesianSimulator
from BayesianSE._plotting import plot_cross_entropies_vs_step, histo_final_cross_entropy
from BayesianSE._statistics import compute_variance, compute_statistics
from BayesianSE._utils import cleaning_convergence
from BayesianSE._noise_models import var_misfreq
from BayesianSE._run_manager import init_new_run, save_metadata, save_results


def run_bayesian_state_estimation(
    molecule: Optional[Any] = None, 
    molecule_type: str = "CaOH", 
    temperature: float = 300, 
    b_field_gauss: float = 3.27, 
    j_max: int = 50,
    rabi_by_j: Union[float, Dict] = 2 * np.pi * 0.001, 
    dephased: bool = False, 
    coherence_time_us: float = 1000.0, 
    is_minus: bool = False,
    false_positive_rate: float = 0.0, 
    false_negative_rate: float = 0.0,
    noise_params: Optional[Dict] = None, 
    seed: Optional[int] = None, 
    laser_miscalibration: Optional[Dict] = None, 
    seed_miscalibration: Optional[int] = None,
    noise_params_estim: Optional[Dict] = None, 
    laser_miscalibration_estim: Optional[Dict] = None,
    pop_fit: Optional[np.ndarray] = None,
    N: int = 100, 
    num_updates: int = 10, 
    block_steps: int = 1, 
    type_block: Optional[str] = None, 
    apply_pumping: bool = False, 
    marginalization: bool = True, 
    false_rates: bool = True,
    save_data: bool = True, 
    only_total: bool = False,
    max_excitation: float = 0.9
) -> Dict[str, Any]:
    """
    Executes a complete Bayesian State Estimation study over N Monte Carlo runs.

    Parameters
    ----------
    molecule : Molecule, optional
        The molecular model object. If None, initialized based on molecule_type.
    molecule_type : str
        Species identifier ('CaH' or 'CaOH').
    temperature : float
        Rotational temperature in Kelvin.
    b_field_gauss : float
        Magnetic field strength in Gauss.
    j_max : int
        Maximum rotational level considered.
    rabi_by_j : float or dict
        Rabi frequency configuration for the measurement pulses.
    dephased : bool
        Whether to include dephasing in the excitation model.
    coherence_time_us : float
        Coherence time for dephasing in microseconds.
    is_minus : bool
        Initial transition direction for the measurement sweep.
    false_positive_rate : float
        Probability of a dark state appearing bright (FPR).
    false_negative_rate : float
        Probability of a bright state appearing dark (FNR).
    noise_params : dict, optional
        Physical noise parameters for the simulator.
    seed : int, optional
        Random seed for the simulation.
    laser_miscalibration : dict, optional
        Systematic calibration errors for the simulator.
    seed_miscalibration : int, optional
        Random seed for miscalibration noise.
    noise_params_estim : dict, optional
        Noise model parameters used by the Bayesian Estimator.
    laser_miscalibration_estim : dict, optional
        Miscalibration model used by the Bayesian Estimator.
    pop_fit : np.ndarray, optional
        Sigmoid fitting parameters for optical pumping efficiencies.
    N : int
        Number of Monte Carlo iterations (independent runs).
    num_updates : int
        Number of global sweeps (or cycles) per run.
    block_steps : int
        Number of measurement repetitions within a single block.
    type_block : str, optional
        The scheduling pattern for measurement pulses.
    apply_pumping : bool
        If True, repumping is applied during the run.
    marginalization : bool
        If True, the estimator integrates likelihoods over the noise distribution.
    false_rates : bool
        If True, the likelihood accounts for FPR and FNR.
    save_data : bool
        Whether to store individual run histories.
    only_total : bool
        Plotting flag for histograms.
    max_excitation : float
        The maximum attainable excitation probability for the pulses.

    Returns
    -------
    results : dict
        A comprehensive dictionary containing statistics, convergence fractions, 
        and raw step data for converged and non-converged runs.
    """

    curves_by_label = {'LM': [], 'PU': [], 'PL': [], 'Other': []}
    variance_by_label = {'LM': [], 'PU': [], 'PL': [], 'Other': []}
    misfrequency_by_label = {'LM': [], 'PU': [], 'PL': [], 'Other': []}

    j_samples = []
    xi_samples = []
    m_samples = []

    color_map = {
        'LM': '#800020',  
        'PU': '#4B0082',  
        'PL': '#DAA520',  
        'Other': '#0C630B'
    }

    Estimator = BayesianStateEstimation(
        model=molecule, 
        temperature=temperature, 
        b_field_gauss=b_field_gauss, 
        j_max=j_max,
        false_positive_rate=false_positive_rate, 
        false_negative_rate=false_negative_rate
    )

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
        marginalization=marginalization
    )

    for run in range(N):
        seed_miscalibration = random.randint(0, 10000)
        print("Run = ", run)
        
        Simulator = BayesianSimulator(
            model=molecule, 
            temperature=temperature, 
            b_field_gauss=b_field_gauss, 
            j_max=j_max, 
            false_positive_rate=false_positive_rate, 
            false_negative_rate=false_negative_rate
        )  

        j_sample = Simulator.history_list[0][1]
        xi_sample = Simulator.history_list[0][3]
        m_sample = Simulator.history_list[0][2]

        j_samples.append(j_sample)  
        xi_samples.append(xi_sample)
        m_samples.append(m_sample)

        Estimator.update_distibution(
            Simulator, 
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

    plot_cross_entropies_vs_step(curves_by_label, color_map, filename="run_CE.svg")

    total_runs, success_rate_total, leftmost_total, success_rate_leftmost = histo_final_cross_entropy(
        curves_by_label, color_map, only_total, filename="run_CEhist.svg"
    )

    stats, all_step, below_step, above_step = compute_statistics(
        curves_by_label, num_updates, block_steps, filename="run_stats.svg"
    )

    var_misfreq(variance_by_label, misfrequency_by_label, filename="run_varmisfreq.svg")

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
    total_filtered, success_rate_filtered, _, _ = histo_final_cross_entropy(
        curves_by_label_filtered, color_map, only_total, filename="run_CEhist_conv.svg"
    )
    stats_filtered, all_step_filtered, below_step_filtered, above_step_filtered = compute_statistics(
        curves_by_label_filtered, num_updates, block_steps, plot=False, filename="run_stats_conv.svg"
    )
    var_misfreq(variance_by_label_filtered, misfrequency_by_label_filtered)

    print("###########################")
    print("### NON-Converging runs ###")
    print("############################")
    total_not_conv, success_rate_not_conv, _, _ = histo_final_cross_entropy(
        curves_by_label_not_converged, color_map, only_total, filename="run_CEhist_nonconv.svg"
    )
    stats_not_conv, all_step_not_conv, below_step_not_conv, above_step_not_conv = compute_statistics(
        curves_by_label_not_converged, num_updates, block_steps, plot=False, filename="run_stats_nonconv.svg"
    )
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

    save_metadata(
        molecule_type, temperature, b_field_gauss, j_max, rabi_by_j, dephased, coherence_time_us, is_minus,
        false_positive_rate, false_negative_rate, noise_params, seed, laser_miscalibration, seed_miscalibration,
        noise_params_estim, laser_miscalibration_estim, pop_fit, N, num_updates,
        block_steps, type_block, apply_pumping, marginalization, false_rates, save_data, only_total, max_excitation
    )
    
    save_results(results)
    
    return results