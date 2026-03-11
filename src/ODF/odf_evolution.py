# File: odf_evolution.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Quantum Dynamics Module for Molecule-Atom Systems.

This module provides tools to simulate the temporal evolution of molecular states
coupled to motional modes and atomic transitions. It includes functions for 
3-level and 2-level molecular systems, handling various Hamiltonians (ODF, BSB, TOT) 
and experimental imperfections like decoherence.
"""

from qutip import *
import numpy as np
from typing import Optional, List, Tuple, Any

from ODF.odf_operators import decay, x_op_3mol_atom, p_op_3mol_atom
from ODF.odf_plotting import plot_atom_mol3_evolution, plot_odf_3mol_data
from ODF.odf_hamiltonians import *


###################################################################################
########################## MOLECULE + MOTION + ION ################################
########################## MOLECULE WITH 3 LEVELS  ################################
###################################################################################



def molec3_atom_evolution(
    psi0: Qobj,
    Ham: str,
    times: np.ndarray,
    times_atom: np.ndarray,
    det: np.ndarray,
    x: Qobj,
    p: Qobj,
    N: int,
    rr: Optional[float] = None,
    rr_mol: Optional[float] = None,
    w_mol: Optional[float] = None,
    rr_atom: Optional[float] = None,
    var_time: bool = True,
    coherence: bool = False,
    return_results: bool = False
) -> Tuple[List[Any], List[float], List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    """
    Simulates the evolution of a 3-level molecule coupled to motion and an atom.

    Parameters
    ----------
    psi0 : Qobj
        Initial state vector or density matrix.
    Ham : str
        Type of Hamiltonian to use ('bsb', 'tot', 'odf').
    times : np.ndarray
        Array of time steps for the molecular evolution.
    times_atom : np.ndarray
        Array of time steps for the atomic readout.
    det : np.ndarray
        List of detunings to iterate over.
    x : Qobj
        Position operator.
    p : Qobj
        Momentum operator.
    N : int
        Number of motional states.
    rr : float, optional
        Rabi rate for the ODF/TOT Hamiltonian.
    rr_mol : float, optional
        Rabi rate for the molecule.
    w_mol : float, optional
        Molecular transition frequency.
    rr_atom : float, optional
        Rabi rate for the atomic readout.
    var_time : bool, optional
        If True, scales pulse duration based on detuning. Default is True.
    coherence : bool, optional
        If True, includes decay operators. Default is False.
    return_results : bool, optional
        If True, returns the full list of solver Result objects. Default is False.

    Returns
    -------
    tuple
        (results, excitation, spin_down_list, spin_up_list, atom_spin_down_list, atom_spin_up_list)
    """
    excitation = []
    spin_down_list = []
    spin_up_list = []
    spin_aux_list = []
    atom_spin_up_list = []
    atom_spin_down_list = []
    results = []

    # Handle decoherence
    coherence_time = 100
    if coherence:
        c_ops = [tensor(decay(coherence_time), qeye(N), qeye(2))]
    else:
        c_ops = []

    # Solver options for high precision
    opts = Options(atol=1e-10, rtol=1e-10, store_states=True)    

    # Precompute expectation operators to improve performance
    n_op = tensor(qeye(3), num(N), qeye(2))
    P_aux = tensor(basis(3, 2) * basis(3, 2).dag(), qeye(N), qeye(2))
    P_down = tensor(basis(3, 1) * basis(3, 1).dag(), qeye(N), qeye(2))
    P_up = tensor(basis(3, 0) * basis(3, 0).dag(), qeye(N), qeye(2))

    e_ops = [n_op, P_aux, P_down, P_up]

    for delta in det:
         
        # Adaptive time scaling: adjusts pulse duration based on the specific detuning
        if var_time:
            time_steps_mol = 50
            t_max = 1.5 * 1e4     # 15ms limit
            loop_min = 1
            loop_max = 4

            # Prevent division by zero for resonance
            abs_delta = np.abs(delta) if np.abs(delta) > 1e-10 else 1e-10

            # Normalize delta to scale the number of loops
            delta_norm = abs_delta / np.max(np.abs(det))
            n_loops = int(np.round(loop_min + (loop_max - loop_min) * delta_norm))

            final_time = min(n_loops * 2 * np.pi / abs_delta, t_max)
            times = np.linspace(0, final_time, time_steps_mol)

            # Rescale molecular Rabi rate based on the final pulse duration
            rr_mol = np.pi / final_time

        # Select the appropriate Hamiltonian
        args = {}
        if Ham == "bsb":
            args = {'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2 * np.pi * w_mol, 'N': N}
            H = H_mol3_atom_bsb
        elif Ham == "tot":
            args = {'rabi_rate': rr, 'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2 * np.pi * w_mol, 'N': N}
            H = H_mol3_atom_tot
        elif Ham == "odf":
            args = {'rabi_rate': rr, 'd': delta, 'N': N}
            H = H_mol3_atom_odf
        else:
            raise ValueError("Invalid Hamiltonian type")

        # Solve Master Equation for the molecule-atom system
        result = mesolve(H, psi0, times, c_ops, e_ops, args=args, options=opts)
        final_state_mol = result.states[-1]

        if return_results:
            results.append(result)
        
        # Store expectation values for the final state
        excitation.append(result.expect[0][-1])
        spin_aux_list.append([result.expect[1][-1]])
        spin_down_list.append([result.expect[2][-1]])
        spin_up_list.append([result.expect[3][-1]])

        # RBR ATOM READOUT: Transfer information from motion to the atom
        args_atom = {'rabi_rate_atom': rr_atom, 'N': N}
        result_atom = mesolve(H_mol3_atom_rsb, final_state_mol, times_atom, [], [], args=args_atom)

        final_state_atom = result_atom.states[-1]

        # Extract atomic populations
        spin_down_atom = [expect(tensor(qeye(3), qeye(N), basis(2, 1) * basis(2, 1).dag()), final_state_atom)]
        spin_up_atom = [expect(tensor(qeye(3), qeye(N), basis(2, 0) * basis(2, 0).dag()), final_state_atom)]

        atom_spin_down_list.append(spin_down_atom)
        atom_spin_up_list.append(spin_up_atom)
    
    return results, excitation, spin_down_list, spin_up_list, atom_spin_down_list, atom_spin_up_list


def odf_change_B(
    b_field_gauss: List[float],
    w_mols: List[float],
    heights: List[float],
    x_data: np.ndarray,
    y_data: np.ndarray,
    motional_ground: Qobj,
    times: np.ndarray,
    times_atom: np.ndarray,
    detunings: np.ndarray,
    N: int,
    rabi_rate: float,
    rabi_rate_molecule: float,
    rabi_rate_atom: float,
    var_time: bool,
    coherence: bool,
    return_results: bool
) -> None:
    """
    Evaluates ODF performance across different magnetic field values.

    Parameters
    ----------
    b_field_gauss : list
        List of magnetic field values in Gauss.
    w_mols : list
        Molecular frequencies corresponding to the B fields.
    heights : list
        Population coefficients for the initial molecular state.
    x_data, y_data : np.ndarray
        Experimental data for comparison/plotting.
    motional_ground : Qobj
        Initial motional state (usually Fock state |0>).
    times, times_atom, detunings : np.ndarray
        Simulation time grids and detuning range.
    N : int
        Motional Hilbert space dimension.
    rabi_rate, rabi_rate_molecule, rabi_rate_atom : float
        Rabi rates for different interactions.
    var_time, coherence, return_results : bool
        Simulation flags passed to molec3_atom_evolution.
    """
    spin_up = basis(3, 0)
    spin_down = basis(3, 1)
    auxiliary_state = basis(3, 2)

    spin_down_atom = basis(2, 1)

    x = x_op_3mol_atom(N)
    p = p_op_3mol_atom(N)

    final_time = times[-1]

    for i in range(len(b_field_gauss)):
        B_field = b_field_gauss[i]
        w_mol = w_mols[i]
        coeff = heights[i]

        print(f"B field = {B_field} G, w_mol = {w_mol*1e6:.2f} Hz, population coefficient = {coeff:.4f}")

        # Construct the initial molecular state as a superposition
        molecule_state = np.sqrt(coeff) * spin_down + np.sqrt(1 - coeff) * auxiliary_state

        # Initialize the full system state: tensor(molecule, motion, atom)
        psi0 = tensor(molecule_state, motional_ground, spin_down_atom) 
        
        # Run evolution for different Hamiltonians to compare results
        results_odf, exc_mol_odf, _, _, _, spin_up_atom_odf = molec3_atom_evolution(
            psi0, "odf", times, times_atom, detunings, x, p, N, 
            rr=rabi_rate, rr_atom=rabi_rate_atom, var_time=var_time, 
            coherence=coherence, return_results=return_results
        )
        
        results_bsb, exc_mol_bsb, _, _, _, spin_up_atom_bsb = molec3_atom_evolution(
            psi0, "bsb", times, times_atom, detunings, x, p, N, 
            rr_mol=rabi_rate_molecule, w_mol=w_mol, rr_atom=rabi_rate_atom, 
            var_time=var_time, coherence=coherence, return_results=return_results
        )
        
        results_tot, exc_mol_tot, _, _, _, spin_up_atom_tot = molec3_atom_evolution(
            psi0, "tot", times, times_atom, detunings, x, p, N, 
            rr=rabi_rate, rr_mol=rabi_rate_molecule, w_mol=w_mol, rr_atom=rabi_rate_atom, 
            var_time=var_time, coherence=coherence, return_results=return_results
        )

        plot_atom_mol3_evolution(
            x_data, y_data, final_time, w_mol, rabi_rate_molecule, rabi_rate, 
            detunings, exc_mol_odf, exc_mol_bsb, exc_mol_tot, 
            spin_up_atom_odf, spin_up_atom_bsb, spin_up_atom_tot, B=B_field
        )


def odf_3mol_data(
    N: int,
    coeff: float,
    motional_ground: Qobj,
    times: np.ndarray,
    times_atom: np.ndarray,
    detunings: np.ndarray,
    rabi_rate: float,
    rabi_rate_atom: float,
    var_time: bool,
    coherence: bool,
    return_results: bool,
    x_data: np.ndarray,
    y_data: np.ndarray
) -> None:
    """
    Simulates and plots ODF data specifically for a 3-level molecule.

    Parameters
    ----------
    N : int
        Motional space cutoff.
    coeff : float
        Population coefficient for the spin-down state.
    motional_ground : Qobj
        Initial state of the motional mode.
    times, times_atom, detunings : np.ndarray
        Simulation grids.
    rabi_rate, rabi_rate_atom : float
        Rabi rates.
    var_time, coherence, return_results : bool
        Control flags.
    x_data, y_data : np.ndarray
        Reference data for plotting.
    """
    spin_down = basis(3, 1)
    auxiliary_state = basis(3, 2)
    spin_down_atom = basis(2, 1)

    # Standard position and momentum operators in the tensor space
    x = tensor(qeye(3), create(N) + destroy(N), qeye(2)) / 2
    p = tensor(qeye(3), 1j * (create(N) - destroy(N)), qeye(2)) / 2

    molecule_state = np.sqrt(coeff) * spin_down + np.sqrt(1 - coeff) * auxiliary_state
    psi0 = tensor(molecule_state, motional_ground, spin_down_atom) 

    _, _, _, _, _, spin_up_atom_odf = molec3_atom_evolution(
        psi0, "odf", times, times_atom, detunings, x, p, N, 
        rr=rabi_rate, rr_atom=rabi_rate_atom, var_time=var_time, 
        coherence=coherence, return_results=return_results
    )

    plot_odf_3mol_data(detunings, spin_up_atom_odf, x_data, y_data)


###################################################################################
########################## MOLECULE + MOTION + ION ################################
###################################################################################



def molec_atom_evolution(
    psi0: Qobj,
    Ham: str,
    times: np.ndarray,
    times_atom: np.ndarray,
    det: np.ndarray,
    x: Qobj,
    p: Qobj,
    N: int,
    rr: Optional[float] = None,
    rr_mol: Optional[float] = None,
    w_mol: Optional[float] = None,
    rr_atom: Optional[float] = None
) -> Tuple[List[Any], List[float], List[float], List[float], List[float], List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    """
    Evolution of a 2-level molecule coupled to motion and an atom readout.

    Parameters
    ----------
    psi0 : Qobj
        Initial state.
    Ham : str
        Hamiltonian type ('bsb', 'tot', 'odf').
    times, times_atom, det : np.ndarray
        Time and frequency arrays.
    x, p : Qobj
        Operators.
    N : int
        Hilbert space size.
    rr, rr_mol, w_mol, rr_atom : float, optional
        Physical parameters for the interaction.

    Returns
    -------
    tuple
        (results, excitation, prob, x_vals, p_vals, spin_down_list, spin_up_list, atom_spin_down_list, atom_spin_up_list)
    """
    x_vals, p_vals, prob, excitation = [], [], [], []
    spin_down_list, spin_up_list = [], []
    atom_spin_up_list, atom_spin_down_list = [], []

    opts = Options(atol=1e-10, rtol=1e-8)
    results = []

    for delta in det:
        init_psi = psi0
        
        # Select Hamiltonian logic for 2-level molecule
        if Ham == "bsb":
            args = {'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2*np.pi * w_mol, 'N': N}
            result = mesolve(H_mol_atom_bsb, init_psi, times, [], [], args=args, options=opts)
        elif Ham == "tot":
            args = {'rabi_rate': rr, 'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2*np.pi * w_mol, 'N': N}
            result = mesolve(H_mol_atom_tot, init_psi, times, [], [], args=args, options=opts)
        elif Ham == "odf":
            args = {'rabi_rate': rr, 'd': delta, 'N': N}
            result = mesolve(H_mol_atom_odf, init_psi, times, [], [], args=args, options=opts)
        else:
            print("Invalid type")

        results.append(result)
        final_state_mol = result.states[-1]

        # Calculate state population after tracing out subsystems
        final_state_mol_p0 = final_state_mol.ptrace(1)[0, 0]
        norm = np.abs(final_state_mol_p0)**2
        prob.append(1 - norm)

        # Motional excitation calculation
        excitation_prob = expect(tensor(qeye(2), num(N), qeye(2)), final_state_mol)
        excitation.append(excitation_prob)

        # Phase space coordinates
        x_vals.append(expect(x, final_state_mol))
        p_vals.append(expect(p, final_state_mol))

        # Molecular spin populations
        spin_down_list.append([expect(tensor(basis(2, 1) * basis(2, 1).dag(), qeye(N), qeye(2)), final_state_mol)])
        spin_up_list.append([expect(tensor(basis(2, 0) * basis(2, 0).dag(), qeye(N), qeye(2)), final_state_mol)])

        # Atomic readout simulation
        args_atom = {'rabi_rate_atom': rr_atom, 'N': N}
        result_atom = mesolve(H_mol_atom_rsb, final_state_mol, times_atom, [], [], args=args_atom)
        final_state_atom = result_atom.states[-1]

        atom_spin_down_list.append([expect(tensor(qeye(2), qeye(N), basis(2, 1) * basis(2, 1).dag()), final_state_atom)])
        atom_spin_up_list.append([expect(tensor(qeye(2), qeye(N), basis(2, 0) * basis(2, 0).dag()), final_state_atom)])

    return results, excitation, prob, x_vals, p_vals, spin_down_list, spin_up_list, atom_spin_down_list, atom_spin_up_list


def molec_evolution(
    psi0: Qobj,
    Ham: str,
    times: np.ndarray,
    det: np.ndarray,
    x: Qobj,
    p: Qobj,
    N: int,
    rr: Optional[float] = None,
    rr_mol: Optional[float] = None,
    w_mol: Optional[float] = None
) -> Tuple[List[float], List[float], List[float], List[float], List[List[float]], List[List[float]]]:
    """
    Evolution of a 2-level molecule coupled to motion (no atom).

    Parameters
    ----------
    psi0 : Qobj
        Initial state.
    Ham : str
        Hamiltonian type.
    times, det : np.ndarray
        Time and detuning grids.
    x, p : Qobj
        Operators.
    N : int
        Hilbert space size.
    rr, rr_mol, w_mol : float, optional
        Interaction parameters.

    Returns
    -------
    tuple
        (excitation, prob, x_vals, p_vals, spin_down_list, spin_up_list)
    """
    x_vals, p_vals, prob, excitation = [], [], [], []
    spin_down_list, spin_up_list = [], []

    for delta in det:
        init_psi = psi0
        if Ham == "bsb":
            args = {'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2*np.pi * w_mol, 'N': N}
            result = mesolve(H_mol_bsb, init_psi, times, [], [], args=args)
        elif Ham == "tot":
            args = {'rabi_rate': rr, 'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2*np.pi * w_mol, 'N': N}
            result = mesolve(H_mol_tot, init_psi, times, [], [], args=args)
        elif Ham == "odf":
            args = {'rabi_rate': rr, 'd': delta, 'N': N}
            result = mesolve(H_mol_odf, init_psi, times, [], [], args=args)
        else:
            print("Invalid type")

        final_state_mol = result.states[-1]

        # Tracing out to calculate ground state population
        final_state_mol_p0 = final_state_mol.ptrace(1)[0, 0]
        prob.append(1 - np.abs(final_state_mol_p0)**2)

        # Observables
        excitation.append(expect(tensor(qeye(2), num(N)), final_state_mol))
        x_vals.append(expect(x, final_state_mol))
        p_vals.append(expect(p, final_state_mol))

        spin_down_list.append([expect(tensor(basis(2, 1) * basis(2, 1).dag(), qeye(N)), final_state_mol)])
        spin_up_list.append([expect(tensor(basis(2, 0) * basis(2, 0).dag(), qeye(N)), final_state_mol)])
    
    return excitation, prob, x_vals, p_vals, spin_down_list, spin_up_list