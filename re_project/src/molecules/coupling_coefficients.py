import matplotlib.pyplot as plt
import numpy as np

from saving import save_figure_in_images


def plot_couplings(
    mo1,
    mo2,
    j_max,
    molecule_type="CaOH",  
    xi1_filter=False,
    xi2_filter=False,
    save_name="coupling_plot.svg"
):

    if molecule_type.lower() == "caoh":
        j_step = 5
    elif molecule_type.lower() == "cah":
        j_step = 2
    else:
        raise ValueError(
            f"Unknown molecule_type '{molecule_type}'. "
            "Supported types: 'CaOH', 'CaH'."
        )

    if xi1_filter is False and xi2_filter is False:
        manifold_label = "Upper sub-manifold"
    elif xi1_filter is True and xi2_filter is True:
        manifold_label = "Lower sub-manifold"
    elif xi1_filter is False and xi2_filter is True:
        manifold_label = "Cross manifold (Upper to Lower)"
    elif xi1_filter is True and xi2_filter is False:
        manifold_label = "Cross manifold (Lower to Upper)"
    else:
        manifold_label = "Custom manifold"

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    cmap = plt.get_cmap("tab20")
    color_index = 0

    for j_idx, j_val in enumerate(range(0, j_max + 1, j_step)):

        transitions_dm1 = mo1.transition_df[mo1.transition_df["j"] == j_val]
        filtered_dm1 = transitions_dm1[
            (transitions_dm1["xi1"] == xi1_filter) &
            (transitions_dm1["xi2"] == xi2_filter)
        ].copy()

        transitions_dm2 = mo2.transition_df[mo2.transition_df["j"] == j_val]
        filtered_dm2 = transitions_dm2[
            (transitions_dm2["xi1"] == xi1_filter) &
            (transitions_dm2["xi2"] == xi2_filter)
        ].copy()

        if filtered_dm1.empty and filtered_dm2.empty:
            continue

        coupling_dm1 = filtered_dm1["coupling"].to_numpy()
        m_init_dm1 = filtered_dm1["m1"].to_numpy()

        coupling_dm2 = filtered_dm2["coupling"].to_numpy()
        m_init_dm2 = filtered_dm2["m1"].to_numpy()

        color = cmap(color_index % 10)
        color_index += 1

        axs[0].plot(
            m_init_dm1, coupling_dm1,
            color=color, linestyle='-', marker='^', markersize=3.5
        )
        axs[0].plot(
            m_init_dm2, coupling_dm2,
            label=fr"J={j_val}",
            color=color, linestyle='--', marker='^', markersize=3.5
        )

        axs[1].plot(
            m_init_dm1, np.abs(coupling_dm1),
            color=color, linestyle='-', marker='^', markersize=3.5
        )
        axs[1].plot(
            m_init_dm2, np.abs(coupling_dm2),
            label=fr"J={j_val}",
            color=color, linestyle='--', marker='^', markersize=3.5
        )

    fig.suptitle(
        f"{molecule_type} - Coupling Coefficients - {manifold_label}",
        fontsize=14,
        weight='bold'
    )

    axs[0].set_title("Signed Coupling", fontsize=18)
    axs[1].set_title("Absolute Coupling", fontsize=18)

    for ax in axs:
        ax.set_xlabel(r"$m_F^\mathrm{init}$", fontsize=16)
        ax.grid(True, linestyle=':', alpha=0.6)

    axs[0].set_ylabel("Coupling coefficient", fontsize=16)

    axs[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(right=0.82)

    save_figure_in_images(fig, save_name)

    plt.show()