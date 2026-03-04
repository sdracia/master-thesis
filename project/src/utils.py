"""
Module for visualizing molecular state distributions and transitions.

This module provides functions to:
- Identify main peaks in molecular spectra.
- Plot figures of merit (FOM) over magnetic field and J manifolds.
- Plot transitions between molecular states.
- Plot state distributions and heatmaps of populations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks
from scipy.constants import h, physical_constants
from numpy import pi
from saving import save_figure_in_images

# Constants
mu_N = physical_constants["nuclear magneton"][0]
gI = physical_constants["proton g factor"][0]

plt.rcParams['font.family'] = 'DejaVu Sans'


def find_main_peaks(freq: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """
    Find the two dominant peaks in a spectrum.

    Parameters
    ----------
    freq : array-like
        Array of frequencies.
    prob : array-like
        Array of excitation probabilities.

    Returns
    -------
    tuple
        Frequencies of the two main peaks, sorted ascending.
    """
    freq = np.asarray(freq)
    prob = np.asarray(prob)

    # Identify all local maxima
    peaks, _ = find_peaks(prob)

    if len(peaks) < 2:
        raise ValueError("Less than 2 peaks found. Check the data")

    # Select the two largest peaks by height
    top_two_idx = peaks[np.argsort(prob[peaks])[-2:]]
    peak_freqs = freq[top_two_idx]
    peak_freqs.sort()

    return peak_freqs[0], peak_freqs[1]


def fom(molecule, states, b_start: float, b_stop: float, j_start: int, j_stop: int, title: str, savetext: str) -> None:
    """
    Plot figure of merit (FOM) vs magnetic field B and J manifold.

    Parameters
    ----------
    molecule : object
        Molecule object with gj, cij_khz, and j_max attributes.
    states : object
        Object representing state populations with method j_distribution().
    b_start : float
        Starting magnetic field (G).
    b_stop : float
        Ending magnetic field (G).
    j_start : int
        Starting J manifold.
    j_stop : int
        Ending J manifold.
    title : str
        Plot title.
    savetext : str
        File name to save figure.
    """
    gj = molecule.gj
    cij = molecule.cij_khz

    num_points = int(np.abs(b_stop - b_start) * 100)
    B_values = np.linspace(b_start, b_stop, num_points)
    J_values = np.linspace(j_start, j_stop + 1, num_points)
    B, J = np.meshgrid(B_values, J_values)

    cb = mu_N * B * 1e-4 / h / 1e3  # in kHz

    x = 0.5 * np.sqrt(cij**2 * ((J + 0.5) ** 2) + (-cb * (gj - gI))**2)
    y = -cb * (gj - gI) / 2
    h0 = 1 - np.abs(y / x)
    F = 1 - h0

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

    # FOM contour
    ax0 = fig.add_subplot(gs[0])
    c = ax0.contourf(J, B, F, 200, cmap='plasma', vmin=0, vmax=1)
    ax0.set_ylabel(r'$B$ (G)', fontsize=25)
    ax0.tick_params(labelbottom=False)
    ax0.set_xlim((j_start, j_stop))
    ax0.set_title(title, fontsize=28)
    ax0.text(molecule.j_max * 0.7, 28, "Paschen-Back regime", color="black", ha="left", fontsize=21)
    ax0.text(molecule.j_max * 0.7, 2, "Zeeman regime", color="white", ha="left", fontsize=21)

    # Population bar chart
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    j_vals = np.arange(molecule.j_max + 1)
    ax1.bar(j_vals, states.j_distribution(), color="navy")
    ax1.set_xlabel(r"$J$", fontsize=25)
    ax1.set_ylabel("Population", fontsize=25)
    ax1.text(molecule.j_max * 0.7, 0.9 * max(states.j_distribution()), r"$T = 300$ K", fontsize=21)
    ax1.set_xlim((j_start, j_stop))
    plt.tick_params(axis='both', which='major', labelsize=25)

    plt.tight_layout()
    save_figure_in_images(fig, savetext)
    plt.show()


def plot_transitions(molecule, caoh: bool, title: str, savetext: str, text: bool = True) -> None:
    """
    Plot signature, penultimate, and sub-manifold splitting transitions per J manifold.

    Parameters
    ----------
    molecule : object
        Molecule with transition_df, gj_list, cij_list, j_max, and cb_khz.
    caoh : bool
        If True, subsample transitions for CaOH visualization.
    title : str
        Figure title.
    savetext : str
        File name to save figure.
    text : bool, optional
        Whether to annotate bars with numeric values.
    """
    # Calculate signature and penultimate transitions
    signature_transitions = np.abs(np.array([molecule.transition_df.loc[molecule.transition_df["j"]==j].iloc[0]["energy_diff"] for j in range(1, molecule.j_max + 1)]))
    penultimate_transitions = np.abs(np.array([molecule.transition_df.loc[molecule.transition_df["j"]==j].iloc[1]["energy_diff"] for j in range(1, molecule.j_max + 1)]))

    # Sub-manifold splitting
    sub_manifold_splitting = []
    for i, j in enumerate(range(1, molecule.j_max + 1)):
        gj = molecule.gj_list[i] if molecule.gj_list != [] else molecule.gj
        cij = molecule.cij_list[i] if molecule.cij_list != [] else molecule.cij_khz
        x = 0.5 * np.sqrt(cij**2 * ((j + 0.5)**2) + (-molecule.cb_khz * (gj - gI))**2)
        sub_manifold_splitting.append(2*x)
    sub_manifold_splitting = np.array(sub_manifold_splitting)

    J = np.arange(1, molecule.j_max + 1)
    bar_width = 0.2

    # Subsample for CaOH
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
            ax.text(J[i] - bar_width, signature_transitions[i] + 1, f'{signature_transitions[i]:.2f}', ha='center', va='bottom', color='red', rotation=90)
            ax.text(J[i], penultimate_transitions[i] + 1, f'{penultimate_transitions[i]:.2f}', ha='center', va='bottom', color='blue', rotation=90)
            ax.text(J[i] + bar_width, sub_manifold_splitting[i] + 1, f'{sub_manifold_splitting[i]:.2f}', ha='center', va='bottom', color='green', rotation=90)

    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    save_figure_in_images(fig, savetext)
    plt.show()


def plot_state_dist(molecule, j: int) -> None:
    """
    Plot the Zeeman energy of all states in a J manifold.

    Parameters
    ----------
    molecule : object
        Molecule object containing state_df and transition_df.
    j : int
        J manifold to plot.
    """
    states_in_j = molecule.state_df.loc[molecule.state_df["j"] == j]
    transitions_in_j = molecule.transition_df[molecule.transition_df["j"] == j]

    m = states_in_j["m"].to_numpy()
    energies = states_in_j["zeeman_energy_khz"].to_numpy()
    state_dist = states_in_j["state_dist"].to_numpy()
    state_dist /= np.sum(state_dist)

    fig, ax = plt.subplots(figsize=(12, 8))
    for mi, ei, ci in zip(m, energies, state_dist):
        ax.hlines(ei, mi - 0.3, mi + 0.3, colors=plt.cm.plasma(ci), linewidth=3)

    ax.set_xlabel("m")
    ax.set_ylabel("Zeeman energy (kHz)")
    ax.set_title(f"Zeeman energies of all states in j={j}, B={molecule.b_field_gauss} G")

    # Colorbar
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap="plasma"), ax=ax)
    cbar.set_label("spin")

    ax.set_xlim(-j-1, j+1)
    ax.set_xticks([i+0.5 for i in range(-j-1, j+1)])

    # Draw arrows for transitions
    for transition in transitions_in_j.itertuples():
        m1, xi1, m2, xi2 = transition.m1, transition.xi1, transition.m2, transition.xi2
        energy1 = molecule.state_df.loc[(molecule.state_df["j"]==j) & (molecule.state_df["m"]==m1) & (molecule.state_df["xi"]==xi1)].zeeman_energy_khz.iloc[0]
        energy2 = molecule.state_df.loc[(molecule.state_df["j"]==j) & (molecule.state_df["m"]==m2) & (molecule.state_df["xi"]==xi2)].zeeman_energy_khz.iloc[0]
        energy_diff = transition.energy_diff
        coupling = transition.coupling
        ax.annotate(
            "",
            xy=(float(m1)-1.0, float(energy1) + float(energy_diff)),
            xytext=(float(m1), float(energy1)),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1)
        )
        ax.text((3*m1 + m2)/4 - 0.5, (3*energy1 + energy2)/4, f"{energy_diff:.2f} kHz", fontsize=8, color="gray")
        ax.text((3*m1 + m2)/4 - 0.5, (3*energy1 + energy2)/4 - 0.9, f"{coupling:.2f}", fontsize=8, color="red")

    plt.show()
    plt.close()


def heatmap_state_pop(dataframe_molecule, j_max: int, normalize: bool = True) -> np.ndarray:
    """
    Create a heatmap visualization of state populations across J and m manifolds.

    Parameters
    ----------
    dataframe_molecule : pd.DataFrame
        Molecule dataframe containing columns ['j','m','xi','state_dist'].
    j_max : int
        Maximum J manifold.
    normalize : bool, optional
        Whether to normalize populations within each J manifold.

    Returns
    -------
    np.ndarray
        2D array representing the population distribution for plotting.
    """
    plt.style.use('default')
    dataframe = dataframe_molecule[["j", "m", "xi", "state_dist"]].copy()

    if normalize:
        for j in range(j_max + 1):
            states_in_j = dataframe.loc[dataframe["j"]==j]
            state_dist = states_in_j["state_dist"].to_numpy()
            state_dist /= np.sum(state_dist)
            dataframe.loc[dataframe["j"]==j, "state_dist"] = state_dist

    df_grouped = dataframe.groupby(['j', 'm']).agg({'xi':'first','state_dist':list}).reset_index()

    # Ensure two elements per xi
    for index, row in df_grouped.iterrows():
        if len(row['state_dist']) == 1:
            if row['xi'] == False:
                df_grouped.at[index,'state_dist'] = [row['state_dist'][0], np.nan]
            elif row['xi'] == True:
                df_grouped.at[index,'state_dist'] = [np.nan, row['state_dist'][0]]

    df = df_grouped[["j","m"]]
    state = df_grouped["state_dist"].tolist()
    sq_array = np.zeros((2*(j_max+1), j_max+1), dtype=object)
    sq_array[:] = np.nan
    list_index = []

    for idx, row in df.iterrows():
        j, m = row['j'], row['m']
        list_index.append([int(j_max + m + 0.5), int(j)])

    for data_idx, idx_tuple in enumerate(list_index):
        sq_array[idx_tuple[0], idx_tuple[1]] = state[data_idx]

    # Plotting
    matrix = sq_array
    vh_cmap = "hsv"
    cmap_shift = 0
    max_weight = 1
    ax_facecolor = '#D3D3D3'
    ax_bkgdcolor = "white"
    ax_color = 'k'
    label_color = "k"
    grid_color = "w"
    grid_bool = True
    ax_labels_bool = True

    plt.figure(figsize=(18, 10))
    ax = plt.gca()
    cmap = matplotlib.colormaps.get_cmap(vh_cmap)
    norm = matplotlib.colors.Normalize(vmin=-pi + np.finfo(float).eps + cmap_shift*2*pi, vmax=pi + cmap_shift*2*pi)

    for (x, y), w in np.ndenumerate(matrix):
        if isinstance(w, float):
            size = 1.0
            face_color = ax_bkgdcolor
            edge_color = ax_bkgdcolor
            rect = plt.Rectangle([x-0.5 - size/2 - j_max, y - size/2], size, size, facecolor=face_color, edgecolor=edge_color)
            ax.add_patch(rect)
        else:
            for i, w_val in enumerate(w):
                if not np.isnan(w_val):
                    size = np.sqrt(abs(w_val)/max_weight)
                    face_color = "blue"
                    edge_color = "blue"
                else:
                    size = 1.0
                    face_color = ax_bkgdcolor
                    edge_color = ax_bkgdcolor
                size_x = size
                size_y = size*0.5
                y_offset = 0.25 if i==0 else -0.25
                rect = plt.Rectangle([x-0.5 - size_x/2 - j_max, y + y_offset - size_y/2], size_x, size_y, facecolor=face_color, edgecolor=edge_color)
                ax.add_patch(rect)

    ax.patch.set_facecolor(ax_facecolor)
    ax.set_aspect("equal", "box")
    ax.set_ylim([-0.5, matrix.shape[1]-0.5])
    ax.set_xlim([-j_max-1, j_max+1])
    ax.set_yticks(np.arange(0, j_max+1))
    ax.set_yticks(np.arange(0, j_max+2)-0.5, minor=True)
    ax.grid(grid_bool, which="minor", color=grid_color)
    ax.tick_params(which="minor", bottom=False, left=False)
    if ax_labels_bool:
        ax.set_xlabel("$m$")
        ax.set_ylabel("$J$")
    ax.xaxis.label.set_color(label_color)
    ax.yaxis.label.set_color(label_color)
    ax.tick_params(axis="x", colors=ax_color)
    ax.tick_params(axis="y", colors=ax_color)
    for spine in ax.spines.values():
        spine.set_color(ax_color)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-j_max-0.5, j_max+1.5, 1))

    plt.style.use(['science', 'notebook', 'ieee'])
    plt.rcParams['font.family'] = 'Times New Roman'

    return matrix