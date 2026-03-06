"""
Bayesian Simulator Module for Molecular State Dynamics.

This module provides the BayesianSimulator class, which acts as the physical 
ground-truth for Bayesian state estimation experiments. It manages the 
stochastic evolution of a single molecule, simulates measurement outcomes 
including experimental noise and imperfections, and tracks the physical state 
history throughout the simulation.
"""

from molecules.molecule import CaH, CaOH
from QLS.state_dist import States
from typing import Optional, Any

from .simulator_utils import j_distribution
from .simulator_likelihood import likelihoods_simulator
from .simulator_dynamics import outcome_simulator
from .simulator_state import state_initialization, new_state_index


class BayesianSimulator:
    """
    Class responsible for simulating the physical behavior of the molecule.
    """

    def __init__(
        self, 
        model: Optional[Any] = None,
        temperature: float = 300, 
        b_field_gauss: float = 3.27, 
        j_max: int = 50, 
        false_positive_rate: float = 0.00,
        false_negative_rate: float = 0.00
    ):
        """
        Initializes the physical simulator.

        The simulator represents the 'real' molecule. At initialization, 
        it selects a starting state based on the provided population distribution 
        (usually the state after pumping and RAP sequences).

        Parameters
        ----------
        model : Molecule, optional
            The molecular model containing state and transition data.
            If None, a CaOH model is created.
        temperature : float, optional
            Rotational temperature in Kelvin for initial state distribution.
            Default is 300.
        b_field_gauss : float, optional
            External magnetic field in Gauss. Default is 3.27.
        j_max : int, optional
            Maximum rotational level J considered. Default is 50.
        false_positive_rate : float, optional
            Probability of a false bright signal (dark count). Default is 0.00.
        false_negative_rate : float, optional
            Probability of a false dark signal. Default is 0.00.
        """
        self.b_field_gauss = b_field_gauss
        self.j_max = j_max

        if model is None:
            model = CaOH.create_molecule_data(
                b_field_gauss=self.b_field_gauss, 
                j_max=self.j_max
            )
        
        self.model = model

        if "state_dist" not in self.model.state_df.columns:
            states1 = States(molecule=self.model, temperature=temperature)

        self.temperature = temperature
        self.fpr = false_positive_rate
        self.fnr = false_negative_rate

        state_index, j_val, m_val, xi_val = self.state_initialization(j_max=self.j_max)

        self.history_list = [[state_index, j_val, m_val, xi_val]]

        self.misfrequency = 0.0    

    j_distribution = j_distribution
    state_initialization = state_initialization
    likelihoods_simulator = likelihoods_simulator
    new_state_index = new_state_index
    outcome_simulator = outcome_simulator