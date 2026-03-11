Bayesian Inference for Molecular Quantum Logic Spectroscopy
===========================================================

This repository contains the code developed for the Master's thesis **"Bayesian Inference for Molecular Quantum Logic Spectroscopy"**, completed as part of the Physics of Data program at the University of Padua. The work was carried out in collaboration with the QCosmo team of the Quantum Optics and Spectroscopy group at the University of Innsbruck.

The code implements theoretical models and numerical simulations to study the rotational state preparation and Bayesian state estimation of polar molecules (CaH and CaOH) in the context of Quantum Logic Spectroscopy (QLS). It includes tools for simulating hyperfine spectra, optical pumping, Rapid Adiabatic Passage (RAP) sequences, optical dipole force and a recursive Bayesian filtering framework for molecular state estimation.

* * *

Overview
--------

Quantum Logic Spectroscopy (QLS) enables precision measurements of molecular systems that are otherwise difficult to probe directly. A key challenge is preparing and identifying the internal rotational state of a single molecule. This repository addresses that challenge through two main components:

*   **Hyperfine Structure**: simulation of the molecular structure and hyperfine spectra
*   **State Preparation**: simulation of optical pumping and RAP sequences to initialize the molecule into a target rotational manifold.
*   **Bayesian State Estimation**: a recursive Bayesian filtering framework that infers the molecular state from a sequence of measurement outcomes, accounting for experimental imperfections such as false positive/negative rates, laser noise, and miscalibration.

The simulations target **CaH** and **CaOH** molecules and use the QuTiP framework for quantum dynamics.

* * *

Repository Structure
--------------------

*   `src/` contains the main source code of the project
*   `notebooks/` contains notebooks that provide a detailed analysis of the framework.
*   `bayesian_runs/` contains output folders from Bayesian Monte Carlo runs
*   `pumping_RAP_data_cah/` contains precomputed pumping + RAP simulation results for CaH
*   `pumping_RAP_data_caoh/` contains precomputed pumping + RAP simulation results for CaOH
*   `odf_json/` contains ODF fit results and processed experimental datasets
*   `molecule_data/` contains generated molecular state and transition CSV files


* * *


Getting Started
---------------

### Requirements

*   Python 3.11.5
*   `numpy`, `scipy`, `matplotlib`, `seaborn` (numerical computing and visualization)
*   `pandas` (data manipulation)
*   `qutip` (quantum dynamics)
*   `lmfit` (curve fitting)
*   `joblib` (parallelization and caching)
*   `imageio` (for animations)
*   `scienceplots` (scientific plot styling)
*   `wigners` (Clebsch-Gordan coefficients)

Install the dependencies with:

    pip install numpy scipy matplotlib seaborn pandas qutip lmfit joblib imageio scienceplots wigners

> **Note:** `scienceplots` requires LaTeX to be installed on your system for font rendering.
> For animations with `matplotlib`, make sure `ffmpeg` is available in your PATH.

### Running the Simulations

The simulations can be run by running the Jupyter notebooks in the `notebooks/` folder, which contains notebooks that provide a detailed analysis of the framework. Some simulations included here were not included in the thesis.


### Reproducing Results

The main analyses presented in the thesis can be reproduced by running the
notebooks contained in the `notebooks/` directory. Each notebook corresponds
to a specific component of the simulation framework:

- molecular structure and hyperfine spectrum generation
- optical pumping and RAP simulations
- optical dipole force simulations
- Bayesian state estimation and Monte Carlo runs

Precomputed datasets required for the Bayesian inference runs are included
in the repository.


* * *

Authors
-------

Andrea Turci  
Physics of Data MSc  
University of Padua

Work carried out in collaboration with the Quantum Optics and Spectroscopy
group at the University of Innsbruck.


Keywords
--------

quantum logic spectroscopy  
molecular ions  
Bayesian inference  
quantum simulation  
CaH  
CaOH


License
-------

This repository is released under the [MIT License](LICENSE).

* * *

Citation
--------

If you use this code in your work, please cite the thesis and this repository:

    Andrea Turci, Bayesian Inference for Molecular Quantum Logic Spectroscopy,
    Master's Thesis, University of Innsbruck, 2025.


If you use this software, please cite it using the following DOI:

[![DOI](https://zenodo.org/badge/920575264.svg)](https://doi.org/10.5281/zenodo.18961430)