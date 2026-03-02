import numpy as np
import matplotlib.pyplot as plt
from qutip import *

from saving import save_figure_in_images




def RAP_plot_internal_and_motional_dynamics(result, times, n_motional, j_val, dm, result_rf=None,
                                            j_plot = None, savetext = "rap_pulse"):
    
    """
     Plot the population dynamics of internal (qubit) and motional (phonon) states
     during a RAP (Rapid Adiabatic Passage) simulation.

     Parameters
     ----------
     result : qutip.Result
         Result object containing the time-evolved quantum states from simulation.
     times : ndarray
         Array of time points corresponding to the evolution steps.
     n_motional : int
         Dimension of the motional Hilbert space (number of phonon states).
     j_val : float
         Total angular momentum quantum number used in the simulation (for labeling).
     dm : int
         Δm quantum number used to characterize the RAP transition.
     result_rf : qutip.Result, optional
         If provided, this is the result of a second simulation (e.g., in the rotating frame)
         that will be plotted with dashed lines for comparison.

     Returns
     -------
     None
         Displays a figure with two subplots:
         - Populations of internal |0⟩ and |1⟩ states over time
         - Mean phonon number ⟨n⟩ over time
    -------------------------------------------------------------------------------
    """

    if j_plot is None:
        do_plot = True
    elif isinstance(j_plot, int):
        do_plot = (j_val == j_plot)
    else:
        do_plot = (j_val in j_plot)

    # Prepare the population matrix
    populations = np.zeros((2, len(times)))
    mean_n = []
    
    # Phonon number operator
    n_op = tensor(qeye(2), num(n_motional))

    for i, state in enumerate(result.states):
        for j in range(2):
            Pj_op = tensor(basis(2, j) * basis(2, j).dag(), qeye(n_motional))
            populations[j, i] = expect(Pj_op, state)

        mean_n.append(expect(n_op, state))

    if result_rf is not None:
        populations_rf = np.zeros((2, len(times)))
        mean_n_rf = []

        for i, state in enumerate(result_rf.states):
            for j in range(2):
                Pj_op_rf = tensor(basis(2, j) * basis(2, j).dag(), qeye(n_motional))
                populations_rf[j, i] = expect(Pj_op_rf, state)

            mean_n_rf.append(expect(n_op, state))

    # --- Plot ---
    if do_plot:

        fig, axs = plt.subplots(1, 2, figsize=(16, 4))

        axs[0].plot(times*1e-3, populations[0], label=r'$|0\rangle$', color='tab:blue', linestyle='-')
        axs[0].plot(times*1e-3, populations[1], label=r'$|1\rangle$', color='tab:orange', linestyle='-')

        axs[0].set_xlabel(r'Time (ms)', fontsize=20)
        axs[0].set_ylabel('Population', fontsize=20)
        axs[0].set_title(
            rf'RAP pulse dynamics, J = {j_val}', 
            fontsize=25
        )
        axs[0].legend(fontsize=16, loc='center left')
        axs[0].grid(True)
        axs[0].tick_params(axis='both', labelsize=20)

        # Phonon number
        axs[1].plot(times*1e-3, mean_n, lw=2, color = "tab:orange")
        axs[1].set_xlabel(r'Time (ms)', fontsize=20)
        axs[1].set_ylabel('<n> (Mean phonon number)', fontsize=20)
        axs[1].set_title('Motional excitation', fontsize=20)
        axs[1].grid(True)

        plt.tight_layout()

        filename = f"{savetext}_j{j_val}.svg"
        save_figure_in_images(fig, filename)

        plt.show()

    last_up_pop = populations[1][-1]

    return last_up_pop



