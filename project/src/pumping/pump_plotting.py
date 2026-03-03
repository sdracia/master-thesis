import numpy as np
import matplotlib.pyplot as plt
from qutip import *

from saving import save_figure_in_images





def plot_internal_and_motional_dynamics(results, times=None, n_internal=None, n_motional=None, title=None, savetext=None, only_pop=False):
    """
    Plots two side-by-side subplots:
    - Heatmap of the internal state populations over time
    - Mean motional excitation (average phonon number) over time

    Parameters:
    - results: either
        * a single QuTiP `Result` object (with `times` provided separately), OR
        * a list of [Result, times] pairs for multiple stages
    - times: array of time points (only used in single-stage mode)
    - n_internal: number of internal states
    - n_motional: dimension of the motional Hilbert space
    - title: optional title for the plots
    - savetext: if given, save the figure with this filename
    """

    if isinstance(results, list):
        # --- MULTI-STAGE CASE ---
        all_times = []
        all_populations = []
        all_mean_n = []
        time_changes = []

        current_time_offset = 0.0

        for idx, (res, t) in enumerate(results):
            
            populations = np.zeros((n_internal, len(t)))

            mean_n = []

            n_op = tensor(qeye(n_internal), num(n_motional))


            for i, state in enumerate(res.states):
                for j in range(n_internal):
                    Pj_op = tensor(basis(n_internal, j) * basis(n_internal, j).dag(), qeye(n_motional))
                    populations[j, i] = expect(Pj_op, state)
                mean_n.append(expect(n_op, state))


            # Append with time shift (to make the stages continuous)
            progressive_time = current_time_offset + t

            if idx < len(results) - 1:
                populations = populations[:, :-1]
                progressive_time = progressive_time[:-1]
                mean_n = mean_n[:-1]


            all_times.append(progressive_time)
            all_populations.append(populations)
            all_mean_n.append(mean_n)
            time_changes.append(progressive_time[-1])


            # update offset
            current_time_offset += t[-1]

        times = np.concatenate(all_times) * 1e-3
        populations = np.concatenate(all_populations, axis=1)
        mean_n = np.concatenate(all_mean_n)

    else:
        # --- SINGLE-STAGE CASE ---
        res = results
        populations = np.zeros((n_internal, len(times)))
        mean_n = []

        n_op = tensor(qeye(n_internal), num(n_motional))

        for i, state in enumerate(res.states):
            for j in range(n_internal):
                Pj_op = tensor(basis(n_internal, j) * basis(n_internal, j).dag(), qeye(n_motional))
                populations[j, i] = expect(Pj_op, state)

            mean_n.append(expect(n_op, state))

        times = times * 1e-3

    if only_pop:


        fig, axs = plt.subplots(figsize=(15, 5))

        # HEATMAP
        im = axs.imshow(populations, aspect='auto', origin='lower', 
                        extent=[times[0], times[-1], -0.5, n_internal - 0.5], cmap='viridis')
        axs.set_xlabel(r'Time ($ms$)', fontsize=25)
        axs.set_ylabel('Internal level index', fontsize=25)
        axs.set_xlim(0, 90)
        xticks = np.arange(times[0], times[-1]+10, 10)
        axs.set_xticks(xticks)
        axs.set_xticklabels([f"{int(x)}" for x in xticks], fontsize=20)

        for t in time_changes[:-1]:
            axs.axvline(t*1e-3, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

        secax = axs.secondary_yaxis('right', functions=(lambda y: y, lambda y: y))
        special_positions = [0, 1, int(n_internal/2)]
        special_labels = ["LM", "PU", "PL"]
        secax.set_yticks(special_positions)
        secax.set_yticklabels(special_labels, fontsize=18)
        secax.minorticks_off()
        secax.tick_params(axis='y', which='both', length=10, direction='inout')

        if title:
            axs.set_title(title, fontsize=28)
        else:
            axs.set_title('Population evolution (internal levels)', fontsize=28)

        cbar = fig.colorbar(im, ax=axs, label="Population", pad =0.12)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label("Population", fontsize=25)

        axs.tick_params(axis='both', pad=10, labelsize=20)

        plt.tight_layout()

        if savetext:
            save_figure_in_images(fig, filename=savetext)

        plt.show()


    else:

        # --- Plot ---
        
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))

        # HEATMAP
        im = axs[0].imshow(populations, aspect='auto', origin='lower', 
                        extent=[times[0], times[-1], -0.5, n_internal - 0.5], cmap='viridis')
        axs[0].set_xlabel(r'Time ($ms$)', fontsize=25)
        axs[0].set_ylabel('Internal level index', fontsize=25)
        xticks = np.linspace(times[0], times[-1], 6)
        axs[0].set_xticks(xticks)
        axs[0].set_xticklabels([f"{int(x)}" for x in xticks], fontsize=20)

        secax = axs[0].secondary_yaxis('right', functions=(lambda y: y, lambda y: y))
        special_positions = [0, 1, int(n_internal/2)]
        special_labels = ["LM", "PU", "PL"]
        secax.set_yticks(special_positions)
        secax.set_yticklabels(special_labels, fontsize=18)
        secax.minorticks_off()
        secax.tick_params(axis='y', which='both', length=10, direction='inout')

        if title:
            axs[0].set_title(title, fontsize=28)
        else:
            axs[0].set_title('Population evolution (internal levels)', fontsize=28)

        cbar = fig.colorbar(im, ax=axs[0], label="Population", pad =0.12)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label("Population", fontsize=25)

        # Phonon number
        axs[1].plot(times, mean_n, color='darkorange', lw=2)
        axs[1].set_xlabel('Time (ms)', fontsize=22)
        axs[1].set_ylabel(r'$\langle n \rangle$ (Mean phonon number)', fontsize=22)
        if title:
            axs[1].set_title(title, fontsize=24)
        else:
            axs[1].set_title('Motional excitation', fontsize=24)
        axs[1].grid(True)

        for ax in axs:
            ax.tick_params(axis='both', pad=10, labelsize=20)

        plt.tight_layout()

        if savetext:
            save_figure_in_images(fig, filename=savetext)

        plt.show()


