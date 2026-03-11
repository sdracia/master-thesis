"""
Bayesian State Estimation and Simulation Sub-package.

This package provides the core infrastructure for molecular state estimation 
through recursive Bayesian filtering. It exposes:
- BayesianStateEstimation: The primary class for maintaining and updating 
  probability distributions (beliefs) of molecular populations.
- BayesianSimulator: The physical simulator used to generate ground-truth 
  molecular behavior and stochastic measurement outcomes.
"""

from .estimator import BayesianStateEstimation
from .simulator import BayesianSimulator

__all__ = [
    "BayesianStateEstimation",
    "BayesianSimulator",
]