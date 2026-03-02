import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from pathlib import Path
from scipy.signal import find_peaks

from scipy.constants import h, physical_constants
mu_N = physical_constants["nuclear magneton"][0]
gI = physical_constants["proton g factor"][0]

from numpy import pi
plt.rcParams['font.family'] = 'DejaVu Sans'

from saving import save_figure_in_images



# import numpy as np

# def apply_noise(value, noise_type, noise_level, seed=None):
#     """Applies noise to a given value with seed control."""

#     rng = np.random.default_rng(seed)  # Se seed è None, otteniamo fluttuazioni casuali

#     if noise_type == "rel_uniform":
#         error = noise_level * np.abs(value) * rng.uniform(-1, 1)

#     elif noise_type in {"rel_gaussian", "rel_normal"}:
#         error = noise_level * np.abs(value) * rng.normal(0, 1)

#     elif noise_type == "abs_uniform":
#         error = noise_level * rng.uniform(-1, 1)

#     elif noise_type in {"abs_gaussian", "abs_normal"}:
#         error = noise_level * rng.normal(0, 1)

#     elif noise_type == "value":
#         error = noise_level

#     else:
#         raise ValueError(f"Tipo di rumore non riconosciuto: {noise_type}")

#     return value + error




def find_main_peaks(freq, prob):
    """
    Find the two main peaks in the spectrum.

    - Identifies all local maxima in the excitation curve.
    - Selects the two main peaks.
    - Sorts the two peaks based on frequency (from lowest to highest).

    Parameters:
    freq : array-like
        Array of frequencies.
    prob : array-like
        Array of excitation probabilities.

    Returns:
    tuple of two floats
        The frequencies of the two main peaks sorted in ascending order.
    """
    freq = np.asarray(freq)
    prob = np.asarray(prob)
    
    peaks, _ = find_peaks(prob)
    
    if len(peaks) < 2:
        raise ValueError("Less than 2 peaks found. Check the data")

    peak_indices = peaks[np.argsort(prob[peaks])[-2:]] 

    peak_frequencies = freq[peak_indices]

    peak_frequencies.sort()

    return peak_frequencies[0], peak_frequencies[1]





####################################################################
################### PLOTTING FUNCTIONS #############################
####################################################################



def fom(molecule, states, b_start, b_stop, j_start, j_stop, title, savetext):

    gj = molecule.gj
    cij = molecule.cij_khz

    num_points = np.abs(b_stop-b_start)*100
    B_values = np.linspace(b_start, b_stop, num_points) 
    J_values = np.linspace(j_start, j_stop+1, num_points)

    B, J = np.meshgrid(B_values, J_values)

    cb = mu_N * B * 1e-4 / h / 1e3

    x = 1 / 2 * np.sqrt(cij**2 * ((J + 1 / 2) ** 2) + (- cb * (gj - gI)) ** 2)
    y = - cb * (gj - gI) / 2 

    h0 = 1 - np.abs(y/x)

    F = 1 - h0

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax0 = fig.add_subplot(gs[0])
    c = ax0.contourf(J, B, F, 200, cmap='plasma', vmin=0, vmax=1)
    # cb = fig.colorbar(c, ax=ax0)
    # cb.set_label(r'$\mathcal{F}(J,B)$')

    ax0.set_ylabel(r'$B$ (G)', fontsize=25)
    ax0.tick_params(labelbottom=False)
    ax0.text(molecule.j_max * 0.7, 28, "Paschen-Back regime", color="black", ha="left", fontsize=21)
    ax0.text(molecule.j_max * 0.7, 2, "Zeeman regime", color="white", ha="left", fontsize=21)
    ax0.set_xlim((j_start, j_stop))
    ax0.set_title(title, fontsize=28)


    j_vals = np.arange(molecule.j_max + 1)
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.bar(j_vals, states.j_distribution(), color="navy")
    ax1.set_xlabel(r"$J$", fontsize=25)
    ax1.set_ylabel("Population", fontsize=25)
    ax1.text(molecule.j_max * 0.7, 0.9 * max(states.j_distribution()), r"$T = 300$ K", fontsize=21)
    ax1.set_xlim((j_start, j_stop))

    plt.tick_params(axis='both', which='major', labelsize=25)

    plt.tight_layout()
    save_figure_in_images(fig, savetext)
    plt.show()




def plot_transitions(molecule, caoh, title, savetext, text = True):
    
    # I analyze the following transitions at different Js, and in different regimes of B:
    # - Signature transition: it's the target transition in the molecule
    # - Penultimate transition: it's the transition at m+1 to the right of the signature transition
    # - Sub-manifold splitting: it's the splitting of the manifold between the states at $\xi = -$ and $\xi = +$ 
    

    signature_transitions =   np.abs(np.array([molecule.transition_df.loc[molecule.transition_df["j"]==j].iloc[0]["energy_diff"] for j in range(1,molecule.j_max+1)]))
    penultimate_transitions = np.abs(np.array([molecule.transition_df.loc[molecule.transition_df["j"]==j].iloc[1]["energy_diff"] for j in range(1,molecule.j_max+1)]))
    sub_manifold_splitting = []

    for i,j in enumerate(range(1, molecule.j_max + 1)):

        gj = molecule.gj_list[i] if molecule.gj_list != [] else molecule.gj
        cij = molecule.cij_list[i] if molecule.gj_list != [] else molecule.cij_khz

        x = 1 / 2 * np.sqrt(cij**2 * ((j + 1 / 2) ** 2) + (- molecule.cb_khz * (gj - gI)) ** 2)

        sub_manifold_splitting.append(2*x)

    sub_manifold_splitting = np.array(sub_manifold_splitting)

    J = np.arange(1, molecule.j_max + 1)

    bar_width = 0.2 

    if caoh:
        indices = np.arange(0, len(J), 4)
        J = J[indices]
        signature_transitions = signature_transitions[indices]
        penultimate_transitions = penultimate_transitions[indices]
        sub_manifold_splitting = sub_manifold_splitting[indices]
        bar_width *= 4


    fig, ax = plt.subplots(figsize=(12, 7.5)) 

    ax.bar(J - bar_width, signature_transitions, width=bar_width, color='red', label='Signature transition')
    ax.bar(J, penultimate_transitions, width=bar_width, color='blue', label='Penultimate transition')
    ax.bar(J + bar_width, sub_manifold_splitting, width=bar_width, color='green', label='Sub-manifold splitting')

    ax.set_xlabel('J manifold', fontsize=25)
    ax.set_ylabel('Frequency (kHz)', fontsize=25)
    ax.set_title(title, fontsize=28)
    ax.set_xticks(J)
    ax.legend(fontsize=25)

    if text:
        for i in range(len(J)):
            ax.text(J[i] - bar_width, signature_transitions[i] + 1, f'{signature_transitions[i]:.2f}',
                    ha='center', va='bottom', color='red', rotation=90)
            ax.text(J[i], penultimate_transitions[i] + 1, f'{penultimate_transitions[i]:.2f}',
                    ha='center', va='bottom', color='blue', rotation=90)
            ax.text(J[i] + bar_width, sub_manifold_splitting[i] + 1, f'{sub_manifold_splitting[i]:.2f}',
                    ha='center', va='bottom', color='green', rotation=90)
    
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    save_figure_in_images(fig, savetext)  
    plt.show()



def plot_state_dist(molecule, j):

    states_in_j = molecule.state_df.loc[molecule.state_df["j"] == j]
    transitions_in_j = molecule.transition_df[molecule.transition_df["j"] == j]
    m = states_in_j["m"].to_numpy()
    energies = states_in_j["zeeman_energy_khz"].to_numpy()

    state_dist = states_in_j["state_dist"].to_numpy()
    # print(state_dist)
    state_dist = state_dist / np.sum(state_dist)
    print(state_dist)



    # spin_up = states_in_j["spin_up"].to_numpy()
    # spin_down = states_in_j["spin_down"].to_numpy()
    colors = state_dist

    fig, ax = plt.subplots(figsize=(12, 8))
    for mi, ei, ci in zip(m, energies, colors):
        ax.hlines(ei, mi - 0.3, mi + 0.3, colors=plt.cm.plasma(ci), linewidth=3)

    ax.set_xlabel("m")
    ax.set_ylabel("Zeeman energy (kHz)")
    ax.set_title(f"Zeeman energies of all states in j={j}, B={molecule.b_field_gauss} G")

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap="plasma"), ax=ax)
    cbar.set_label("spin")

    ax.set_xlim(-j-1, j+1)
    ax.set_xticks([i+0.5 for i in range(-j - 1, j + 1)])


    # plot the difference between neibouring states on arrows conecting them
    for transition in transitions_in_j.itertuples():
        m1 = transition.m1
        xi1 = transition.xi1
        energy1 = molecule.state_df.loc[(molecule.state_df["j"] == j) & (molecule.state_df["m"] == m1) & (molecule.state_df["xi"] == xi1)].iloc[0].zeeman_energy_khz
        m2 = transition.m2
        xi2 = transition.xi2
        energy2 = molecule.state_df.loc[(molecule.state_df["j"] == j) & (molecule.state_df["m"] == m2) & (molecule.state_df["xi"] == xi2)].iloc[0].zeeman_energy_khz
        energy_diff = transition.energy_diff
        coupling = transition.coupling
        ax.annotate(
            "", 
            xy=(float(m1)-1.0, float(energy1) + float(energy_diff)), 
            xytext=(float(m1), float(energy1)),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1)
        )
        # add the energy difference as text on the arrow
        ax.text((3*m1 + m2) / 4.0 -0.5, (3*energy1 + energy2) / 4.0, f"{energy_diff:.2f} kHz", fontsize=8, color="gray")
        # add the coupling strength as text on the arrow
        ax.text((3*m1 + m2) / 4.0 -0.5, (3*energy1 + energy2) / 4.0 - 0.9, f"{coupling:.2f}", fontsize=8, color="red")

    plt.show()
    plt.close()




def heatmap_state_pop(dataframe_molecule, j_max, normalize = True):

    plt.style.use('default')
    
    dataframe = dataframe_molecule[["j", "m", "xi", "state_dist"]].copy()

    ## Renormalize
    if normalize:
        for j in range(j_max + 1):
            states_in_j = dataframe.loc[dataframe["j"] == j]

            state_dist = states_in_j["state_dist"].to_numpy()
            state_dist = state_dist / np.sum(state_dist)

            dataframe.loc[dataframe["j"] == j, "state_dist"] = state_dist

    # print(dataframe["state_dist"].to_numpy())


    df_grouped = dataframe.groupby(['j', 'm']).agg({'xi': 'first', 'state_dist': list}).reset_index()
    
    for index, row in df_grouped.iterrows():
        if len(row['state_dist']) == 1:  # Se la lista contiene un solo valore
            if row['xi'] == False:
                df_grouped.at[index, 'state_dist'] = [row['state_dist'][0], np.nan]  # Aggiungi NaN come secondo elemento
            elif row['xi'] == True:
                df_grouped.at[index, 'state_dist'] = [np.nan, row['state_dist'][0]]  # Aggiungi NaN come primo elemento
    
    
    df = df_grouped[["j", "m"]]
    state = df_grouped["state_dist"].tolist()
    
    
    sq_array = np.zeros((2 * (j_max+1), j_max + 1), dtype=object)    # *2
    sq_array[:] = np.nan
    
    list_index = []
    
    for index, row in df.iterrows():
        j = row['j']
        m = row['m']
        list_index.append([int(j_max + m + 0.5), int(j)])
    
    
    for data_idx, idx_tuple in enumerate(list_index):
        # print(idx_tuple[0], idx_tuple[1])
        # print(state[data_idx])
        sq_array[idx_tuple[0], idx_tuple[1]] = state[data_idx]
    
    
    # for i in range(sq_array.shape[0]):
    #     for j in range(sq_array.shape[1]):
    #         if isinstance(sq_array[i, j], list):
    #             sq_array[i, j] = np.array(sq_array[i, j])  # Converte la lista in un array numpy
    #             sq_array[i, j] = np.where(np.abs(sq_array[i, j]) < 1e-10, 0, sq_array[i, j])  # Applica l'azzeramento
    
    matrix = sq_array


    vh_cmap="hsv"
    cmap_shift=0

    # vh_amp = False
    max_weight = 1
    ax_facecolor='#D3D3D3'
    ax_bkgdcolor="white"
    label_color="k"
    grid_color="w"
    grid_bool = True
    ax_labels_bool=True
    ax_color='k'
    ax_facecolor = '#D3D3D3'
    
    cmap = matplotlib.colormaps.get_cmap(vh_cmap)
    norm = matplotlib.colors.Normalize(
        vmin=-pi + np.finfo(float).eps + cmap_shift * 2 * pi,
        vmax=pi + cmap_shift * 2 * pi,
    )
    

    plt.figure(figsize=(18, 10)) 
    # ax_facecolor = '#D3D3D3'
    # # ax = ax if ax is not None else plt.gca()
    ax = plt.gca()
    # ax.patch.set_facecolor(ax_facecolor)
    # ax.set_aspect("equal", "box")

    
    
    for (x, y), w in np.ndenumerate(matrix):
        # print(x,y,w)
        var = isinstance(w, float)
        # print(var)
    
        #single values: alwasy nan
        if var:
            # print("Ok")
            # if not np.isnan(w):
            #     face_color = cmap(norm(np.angle(w)))
            #     edge_color = None
    
            #     if not vh_amp:
            #         # if vh_amp is False, then hinton plot has rectangles area ~ norm squared of amplitude
            #         w_plot = abs(w) ** 2
            #     else:
            #         # else hinton plot has rectangles ~ norm of amplitude
            #         w_plot = abs(w)
            #     size = np.sqrt(w_plot / max_weight)
            # else:
            size = 1.0
            face_color = ax_bkgdcolor
            edge_color = ax_bkgdcolor
    
            # print("one", x - size / 2 - j_max, "and two", y - size / 2)
            rect = plt.Rectangle(
                [x-0.5 - size / 2 - j_max, y - size / 2],
                size,
                size,
                facecolor=face_color,
                edgecolor=edge_color,
            )
            ax.add_patch(rect)
        else:
            w_false = w[0]
            w_true = w[1]
    
            if not np.isnan(w_false):
                face_color = "blue"
                edge_color = "blue"
    
                # if not vh_amp:
                #     # if vh_amp is False, then hinton plot has rectangles area ~ norm squared of amplitude
                #     w_plot = abs(w_false) ** 2
                # else:
                    # else hinton plot has rectangles ~ norm of amplitude
                w_plot = abs(w_false)
                size = np.sqrt(w_plot / max_weight)
            else:
                size = 1.0
                face_color = ax_bkgdcolor
                edge_color = ax_bkgdcolor
    
            size_x = size
            size_y = size*0.5
            rect = plt.Rectangle(
                [x-0.5 - size_x / 2 - j_max, y + 0.25 - size_y / 2],
                size_x,
                size_y,
                facecolor=face_color,
                edgecolor=edge_color,
            )
            ax.add_patch(rect)
    
            if not np.isnan(w_true):
                face_color = "blue"
                edge_color = "blue"
    
                # if not vh_amp:
                #     # if vh_amp is False, then hinton plot has rectangles area ~ norm squared of amplitude
                #     w_plot = abs(w_true) ** 2
                # else:
                    # else hinton plot has rectangles ~ norm of amplitude
                w_plot = abs(w_true)
                size = np.sqrt(w_plot / max_weight)
            else:
                size = 1.0
                face_color = ax_bkgdcolor
                edge_color = ax_bkgdcolor
    
            size_x = size
            size_y = size*0.5
            rect = plt.Rectangle(
                [x-0.5 - size_x / 2 - j_max, y - 0.25 - size_y / 2],
                size_x,
                size_y,
                facecolor=face_color,
                edgecolor=edge_color,
            )
            ax.add_patch(rect)
            
    # ax = ax if ax is not None else plt.gca()
    # ax = plt.gca()
    ax.patch.set_facecolor(ax_facecolor)
    ax.set_aspect("equal", "box")
    
    ax.set_ylim([-0.5, matrix.shape[1] - 0.5])
    ax.set_xlim([-j_max-1, j_max+1])
    ax.set_yticks(np.arange(0, j_max + 1))
    ax.set_yticks(np.arange(0, j_max + 2) - 0.5, minor=True)
    # ax.set_xticks(np.arange(-j_max, j_max + 1))
    
    ax.grid(grid_bool, which="minor", color=grid_color)
    ax.tick_params(which="minor", bottom=False, left=False)
    if ax_labels_bool:
        ax.set_xlabel("$m$")
        ax.set_ylabel("$J$")
    ax.xaxis.label.set_color(label_color)
    ax.yaxis.label.set_color(label_color)
    ax.tick_params(axis="x", colors=ax_color)
    ax.tick_params(axis="y", colors=ax_color)
    ax.spines["left"].set_color(ax_color)
    ax.spines["bottom"].set_color(ax_color)
    ax.spines["right"].set_color(ax_color)
    ax.spines["top"].set_color(ax_color)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-j_max - 0.5, j_max + 1.5, 1))


    plt.style.use(['science', 'notebook', 'ieee'])
    plt.rcParams['font.family'] = 'Times New Roman'

    return matrix