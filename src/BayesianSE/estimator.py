"""
Bayesian State Estimation Module for Molecular Populations.

This module defines the BayesianStateEstimation class, which implements a 
recursive Bayesian filtering framework to estimate the population distribution 
of molecular states. It integrates experimental likelihoods, update rules, 
and pumping dynamics to refine the state vector based on measurement results.
"""

from molecules.molecule import Molecule, CaH, CaOH
from QLS.state_dist import States
from typing import Optional, Any

# Internal utility imports for Bayesian logic
from .estimator_utils import init_prior, stop_condition, cross_entropy
from .estimator_likelihood import likelihoods_estimator
from .estimator_update import update_distibution, get_next_setting
from .estimator_measurements import measurement_setting
from .estimator_pumping import reset_prior_pumping, within_run_pumping


class BayesianStateEstimation:
    """
    Class for performing Bayesian estimation of molecular state populations.
    """

    def __init__(
        self, 
        model: Optional[Molecule] = None, 
        temperature: float = 300, 
        b_field_gauss: float = 3.27, 
        j_max: int = 50,
        false_positive_rate: float = 0.00,
        false_negative_rate: float = 0.00
    ):
        """
        Initializes the Bayesian Estimator with molecular and experimental parameters.

        Parameters
        ----------
        model : Molecule, optional
            The molecule object containing state and transition data. 
            If None, a CaOH model is created by default.
        temperature : float, optional
            Rotational temperature in Kelvin for the initial distribution. 
            Default is 300.
        b_field_gauss : float, optional
            External magnetic field in Gauss. Default is 3.27.
        j_max : int, optional
            Maximum rotational quantum number J to consider. Default is 50.
        false_positive_rate : float, optional
            The probability of a false detection (dark count). Default is 0.00.
        false_negative_rate : float, optional
            The probability of missing a detection (collection efficiency). 
            Default is 0.00.
        """
        self.b_field_gauss = b_field_gauss
        self.j_max = j_max

        if model is None:
            model = CaOH.create_molecule_data(
                b_field_gauss=self.b_field_gauss, 
                j_max=self.j_max
            )
        
        self.model = model
        self.fpr = false_positive_rate
        self.fnr = false_negative_rate

        if "state_dist" not in self.model.state_df.columns:
            states1 = States(molecule=self.model, temperature=temperature)

        self.temperature = temperature
        
        self.init_prior()
        self.history_list = []

        self.after_pumping_dist = self.model.state_df["state_dist"].copy()


    init_prior = init_prior
    reset_prior_pumping = reset_prior_pumping
    within_run_pumping = within_run_pumping
    likelihoods_estimator = likelihoods_estimator
    measurement_setting = measurement_setting
    update_distibution = update_distibution
    get_next_setting = get_next_setting
    stop_condition = stop_condition
    cross_entropy = cross_entropy