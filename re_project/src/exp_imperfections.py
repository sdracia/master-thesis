import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, physical_constants

from saving import save_figure_in_images
from molecules.molecule import CaH, CaOH, CaOH_dm2, CaH_dm2, Molecule
import QLS.spectrum as spectrum
import QLS.pumping as pumping
import QLS.state_dist as state_dist


mu_N = physical_constants["nuclear magneton"][0]
gI = physical_constants["proton g factor"][0]

####################################################################
################### EXPERIMENTAL IMPERFECTIONS #####################
####################################################################


def apply_noise(value, noise_type, noise_level, seed=None):
    """Applies noise to a given value with seed control.

    Parameters:
        value (float): Original value.
        noise_type (str): Type of noise ('uniform', 'gaussian', or 'normal').
        noise_level (float): Percentage of error.
        seed (int, optional): Seed to make the noise CONSTANT. If None, the noise will be different on each call.

    Returns:
        float: Value with applied noise.
    """

    rng = np.random.default_rng(seed)  # If seed is None, we have random fluctuations.

    if noise_type == "rel_uniform":
        error = noise_level * np.abs(value) * rng.uniform(-1, 1)

    elif noise_type == "rel_gaussian" or noise_type == "rel_normal":
        error = noise_level * np.abs(value) * rng.normal(0, 1)

    elif noise_type == "abs_uniform":
        error = noise_level * rng.uniform(-1, 1)

    elif noise_type == "abs_gaussian" or noise_type == "abs_normal":
        error = noise_level * rng.normal(0, 1)

    elif noise_type == "value":
        error = noise_level

    else:
        error = 0

    return value + error


def false_positive_excitation(
    frequencies: np.ndarray,
    exc_probs: np.ndarray,
    false_positive_rate: float = 0.0,
    type_false_positive: str = "uniform"
) -> np.ndarray:
    """Adds false positive excitation to the spectrum
    
    Args:
        frequencies (np.ndarray): The frequencies of the spectrum
        exc_probs (np.ndarray): The excitation probabilities of the spectrum
        false_positive_rate (float): The false positive rate to add to the spectrum
        type_false_positive (str): The type of false positive to add to the spectrum
    Returns:
        np.ndarray: The excitation probabilities with false positive excitation added
    """
    if false_positive_rate == 0.0:
        return exc_probs

    if type_false_positive == "uniform":
        false_positive_excitation = np.random.uniform(0, false_positive_rate, size=len(frequencies))
    elif type_false_positive == "gaussian":
        false_positive_excitation = np.random.normal(0, false_positive_rate, size=len(frequencies))
    else:
        raise ValueError(f"Unknown type of false positive excitation: {type_false_positive}")

    return exc_probs + false_positive_excitation






def compute_transition_type(mo1, gj, cij, transition_type = "signature"):
    
    if transition_type == "signature":
        transition_exact = np.abs(np.array([mo1.transition_df.loc[mo1.transition_df["j"]==j].iloc[0]["energy_diff"] for j in range(1,mo1.j_max+1)]))

    elif transition_type == "penultimate_upper":
        transition_exact = np.abs(np.array([mo1.transition_df.loc[mo1.transition_df["j"]==j].iloc[1]["energy_diff"] for j in range(1,mo1.j_max+1)]))

    elif transition_type == "penultimate_lower":
        transition_exact = np.abs(np.array([mo1.transition_df.loc[mo1.transition_df["j"]==j].iloc[(2*j+1)*2 - 2]["energy_diff"] for j in range(1,mo1.j_max+1)]))


    elif transition_type == "sub_manifold_splitting":

        transition_exact = []

        for i,j in enumerate(range(1, mo1.j_max + 1)):

            x = 1 / 2 * np.sqrt(cij**2 * ((j + 1 / 2) ** 2) + (- mo1.cb_khz * (gj - gI)) ** 2)

            transition_exact.append(2*x)

        transition_exact = np.array(transition_exact)

    else:
        print("Invalid type. Choose between signature, penultimate_upper, penultimate_lower, and sub_manifold_splitting.")
    
    return transition_exact



def plot_variation_transition(j_max, relative_matrix, parameter, range_param, contours=None, filename="STvsGJcaoh.svg"):
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 11))  # singolo asse

    X, Y = np.meshgrid(np.arange(1, j_max + 1), range_param)
    Z = np.abs(np.array(relative_matrix))

    # Mostra la heatmap come immagine con assi reali
    im = ax.imshow(Z, aspect='auto', origin='lower',
                extent=[1, j_max + 1, range_param[0], range_param[-1]],
                cmap='coolwarm')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Percentage variation', fontsize=25)
    cbar.ax.yaxis.set_tick_params(labelsize=25)

    if contours is not None:
        for level, color in contours:
            contour = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=3)
            ax.clabel(contour, fmt=f"{level}", colors=color, fontsize=25)

    # Etichette e titolo
    ax.set_xlabel('$J$', fontsize=25)


    if parameter == "gj":
        ax.set_ylabel('$g_{j}$', fontsize=25)
        ax.set_title('Signature transition relative variation over $g_{j}$', fontsize=28)
    elif parameter == "cij":
        ax.set_ylabel('$c_{ij}$', fontsize=25)
        ax.set_title('Signature transition relative variation over $c_{ij}$', fontsize=28)
    else:
        print("Invalid parameter. Choose between cij and gj.")

    plt.tick_params(axis='both', which='major', labelsize=25)

    plt.tight_layout()



    if filename is not None:
        save_figure_in_images(fig, filename)
    else:
        save_figure_in_images(fig, "STvsGJcaoh.svg")

    plt.show()




def transition_relative_variation(b_field_gauss, j_max, 
                                  range_param, contours=None, 
                                  parameter = "gj", transition_type = "signature", 
                                  filename="STvsGJcaoh.svg"):

    cij = CaOH.cij_khz
    gj = CaOH.gj

    mo1 = CaOH.create_molecule_data(b_field_gauss=b_field_gauss, j_max=j_max)

    transition_exact = compute_transition_type(mo1, gj, cij, transition_type)



    print('\n')

    transition_matrix = []
    difference_matrix = []
    relative_matrix = []

    for par in range_param:
        if parameter == "gj":
            CaOH.gj = par
        elif parameter == "cij":
            CaOH.cij_khz = par
        else:
            print("Invalid parameter. Choose between cij and gj.")

        mo1 = CaOH.create_molecule_data(b_field_gauss=b_field_gauss, j_max=j_max)

        transition_noised = compute_transition_type(mo1, gj, cij, transition_type)


        diff = transition_noised - transition_exact
        rel_err = (transition_noised - transition_exact) /transition_exact


        difference_matrix.append(diff)
        relative_matrix.append(rel_err) 
        transition_matrix.append(transition_noised)


    transition_matrix = np.array(transition_matrix)

    plot_variation_transition(j_max, relative_matrix, parameter, range_param, contours, filename)

    CaOH.cij_khz = 1.49
    CaOH.gj = -0.036



