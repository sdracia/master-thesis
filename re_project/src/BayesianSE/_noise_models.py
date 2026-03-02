import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os


from ._run_manager import plot_bayesian_run



def compute_sigma(source: dict, key: str, value: float):
    entry = (source or {}).get(key, {})
    sigma_type = entry.get("type")
    level = entry.get("level", 0.0)

    if sigma_type == "abs_gaussian":
        return level

    elif sigma_type == "rel_gaussian":
        return level * abs(value)

    elif sigma_type == "abs_uniform":
        return level / np.sqrt(3)

    elif sigma_type == "rel_uniform":
        return (level * abs(value)) / np.sqrt(3)
    
    else:
        return 0.0



def build_detuning_distributions(
    frequency: float,
    rabi_rate: float,
    laser_miscalibration: dict,
    noise_params: dict,
    num_points: int = 1000,
    num_sigma: float = 5.0
):
    # Even though experimental imperfections can come also from the Rabi rate, we only consider the frequency, since they 
    # are less problematic. The marginalization is performed on frequency only.


    # --- FREQUENCY ---
    sigma_freq = np.sqrt(
        compute_sigma(laser_miscalibration, "frequency", frequency)**2 +
        compute_sigma(noise_params, "frequency", frequency)**2
    )

    freq_grid = np.linspace(
        frequency - num_sigma * sigma_freq,
        frequency + num_sigma * sigma_freq,
        num_points
    )
    
    dx_freq = freq_grid[1] - freq_grid[0]
    freq_pdf = norm.pdf(freq_grid, loc=frequency, scale=sigma_freq)
    freq_probs = freq_pdf * dx_freq
    freq_probs /= freq_probs.sum()  

    return freq_grid, freq_probs





def var_misfreq(variance_by_label, misfreq_by_label, filename="figure_variance_misfreq.svg"):

    final_variance_below = []
    final_variance_above = []
    threshold = -np.log(0.9)

    for label, curves in variance_by_label.items():
        for last_var, cross_entropy in curves:
            if cross_entropy < threshold:
                final_variance_below.append(last_var)
            else:
                final_variance_above.append(last_var)

    max = np.max(final_variance_below + final_variance_above)
    min = np.min(final_variance_below + final_variance_above)
    # len = len(final_variance_below + final_variance_above)

    bins = np.arange(min, max, (max-min)/100)

    name, ext = os.path.splitext(filename)
    filename_category = f"{name}_variance{ext}"

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(final_variance_below, bins=bins, alpha=0.6, color="#1f77b4",
            edgecolor='black', label="Success estimation")
    ax.hist(final_variance_above, bins=bins, alpha=0.6, color="#ff7f0e",
            edgecolor='black', label="Failure estimation")

    ax.set_xlabel("Variance", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ax.set_title("Distribution of the final variance values", fontsize=25)
    ax.legend(fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad = 8)


    fig.tight_layout()

    plot_bayesian_run(fig, filename_category)

    plt.show()


    final_misfreq_below = []
    final_misfreq_above = []
    threshold = -np.log(0.9)

    for label, curves in misfreq_by_label.items():
        for last_mis, cross_entropy in curves:
            if cross_entropy < threshold:
                final_misfreq_below.append(last_mis)
            else:
                final_misfreq_above.append(last_mis)

    max = np.max(final_misfreq_below + final_misfreq_above)
    min = np.min(final_misfreq_below + final_misfreq_above)


    if max == min:
        print("Simulation without miscalibration on frequency")
    else:   
        bins = np.arange(min, max, (max-min)/100)

        name, ext = os.path.splitext(filename)
        filename_category = f"{name}_miscal_freq{ext}"


        fig, ax = plt.subplots(figsize=(7, 4))

        ax.hist(final_misfreq_below, bins=bins, alpha=0.6, color="#1f77b4",
                edgecolor='black', label="Success estimation")
        ax.hist(final_misfreq_above, bins=bins, alpha=0.6, color="#ff7f0e",
                edgecolor='black', label="Failure estimation")

        ax.set_xlabel("Raman frequency miscalibration (Hz)", fontsize=20)
        ax.set_ylabel("Frequency", fontsize=20)
        ax.set_title("Distribution of the miscalibration values", fontsize=25)
        ax.legend(fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad = 8)


        fig.tight_layout()

        name, ext = os.path.splitext(filename)
        filename_category = f"{name}_misfreq{ext}"

        plot_bayesian_run(fig, filename_category)
        
        plt.show()