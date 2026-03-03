import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from ._run_manager import plot_bayesian_run
from saving import save_figure_in_images





def matrix_creation(df, posterior, j_max, normalize = True):

    dataframe = df.copy()
    dataframe["posterior"] = posterior

    ## Renormalize
    if normalize:
        for j in range(j_max + 1):
            states_in_j = dataframe.loc[dataframe["j"] == j]

            state_dist = states_in_j["posterior"].to_numpy()
            state_dist = state_dist / np.sum(state_dist)

            dataframe.loc[dataframe["j"] == j, "posterior"] = state_dist


    df_grouped = dataframe.groupby(['j', 'm']).agg({'xi': 'first', 'posterior': list}).reset_index()
    
    for index, row in df_grouped.iterrows():
        if len(row['posterior']) == 1:  # Se la lista contiene un solo valore
            if row['xi'] == False:
                df_grouped.at[index, 'posterior'] = [row['posterior'][0], np.nan]  # Aggiungi NaN come secondo elemento
            elif row['xi'] == True:
                df_grouped.at[index, 'posterior'] = [np.nan, row['posterior'][0]]  # Aggiungi NaN come primo elemento
    
    
    df = df_grouped[["j", "m"]]
    state = df_grouped["posterior"].tolist()
    
    
    sq_array = np.zeros((2 * (j_max+1), j_max + 1), dtype=object)    # *2
    sq_array[:] = np.nan
    
    list_index = []
    
    for index, row in df.iterrows():
        j = row['j']
        m = row['m']
        list_index.append([int(j_max + m + 0.5), int(j)])
    
    
    for data_idx, idx_tuple in enumerate(list_index):
        sq_array[idx_tuple[0], idx_tuple[1]] = state[data_idx]
    

    matrix = sq_array

    return matrix 


def simulator_state(df, index, j_max):

    dataframe = df.copy()

    dataframe["outcome"] = 0
    dataframe.loc[index, "outcome"] = 1


    df_grouped = dataframe.groupby(['j', 'm']).agg({'xi': 'first', 'outcome': list}).reset_index()
    
    for index, row in df_grouped.iterrows():
        if len(row['outcome']) == 1:  # Se la lista contiene un solo valore
            if row['xi'] == False:
                df_grouped.at[index, 'outcome'] = [row['outcome'][0], np.nan]  # Aggiungi NaN come secondo elemento
            elif row['xi'] == True:
                df_grouped.at[index, 'outcome'] = [np.nan, row['outcome'][0]]  # Aggiungi NaN come primo elemento
    
    
    df = df_grouped[["j", "m"]]
    state = df_grouped["outcome"].tolist()
    
    
    sq_array = np.zeros((2 * (j_max+1), j_max + 1), dtype=object)    # *2
    sq_array[:] = np.nan
    
    list_index = []
    
    for index, row in df.iterrows():
        j = row['j']
        m = row['m']
        list_index.append([int(j_max + m + 0.5), int(j)])
    
    
    for data_idx, idx_tuple in enumerate(list_index):
        sq_array[idx_tuple[0], idx_tuple[1]] = state[data_idx]
    

    matrix = sq_array


    for (x, y), w in np.ndenumerate(matrix):

        var = isinstance(w, float)
        if var:
            continue
        else:
            w_false = w[0]
            w_true = w[1]

            if w_false == 1:
                coordinates = [x, y + 0.25]   
            if w_true == 1:
                coordinates = [x, y - 0.25]


    return coordinates 



def plot_matrix(ax, matrix, coordinate_out, j_max):    


    max_weight = 1
    ax_facecolor='#D3D3D3'
    ax_bkgdcolor="white"
    
    ax_facecolor = '#D3D3D3'
    
    
    
    for (x, y), w in np.ndenumerate(matrix):

        var = isinstance(w, float)
    
        #single values: alwasy nan
        if var:

            size = 1.0
            face_color = ax_bkgdcolor
            edge_color = ax_bkgdcolor
    
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
            

    x_out = coordinate_out[0]
    y_out = coordinate_out[1]

    rect = plt.Rectangle(
                [x_out-0.5 - 1 / 2 - j_max, y_out - 0.5 / 2],
                1,
                0.5,
                facecolor='none',
                edgecolor="red",
            )
    ax.add_patch(rect)
    
    ax.patch.set_facecolor(ax_facecolor)
    ax.set_aspect("equal", "box")
    
    ax.set_ylim([-0.5, matrix.shape[1] - 0.5])




def heatmap_posterior_animation(Estimator, Simulator, molecule, j_max, folder="animations", base_filename="evolution_EwS_"):
    
    Estimator_list = Estimator.history_list
    Simulator_list = Simulator.history_list


    posteriors = [np.array(entry["posterior"]) for entry in Estimator_list]
    
    
    matrices = [
        matrix_creation(df=molecule.state_df, posterior=posterior, j_max=j_max, normalize=False)
        for posterior in posteriors
    ]

    
    indices = []
    coordinates = []
    for i in range(len(Simulator_list[1:])):
        index = Simulator_list[i+1][0]
        indices.append(index)
        coordinate = simulator_state(df=molecule.state_df, index=index, j_max=j_max)
        coordinates.append(coordinate)

    
    fig, ax = plt.subplots(figsize=(20, 13))
    ax_facecolor = '#D3D3D3'

    def update(step):
        ax.clear()
        ax.set_facecolor(ax_facecolor)

        matrix = matrices[step]
        coordinate_out = coordinates[step]
        plot_matrix(ax, matrix, coordinate_out, j_max)

        ax.set_xlim([-j_max - 1, j_max + 1])
        ax.set_yticks(np.arange(0, j_max + 1))
        ax.set_yticks(np.arange(0, j_max + 2) - 0.5, minor=True)
        ax.grid(True, which="minor", color="w")
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlabel("$m$")
        ax.set_ylabel("$J$")
        ax.xaxis.label.set_color("k")
        ax.yaxis.label.set_color("k")
        ax.tick_params(axis="x", colors="k")
        ax.tick_params(axis="y", colors="k")
        for spine in ax.spines.values():
            spine.set_color("k")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(-j_max - 0.5, j_max + 1.5, 1))
        ax.set_title(f"Evolution of Estimator Posterior with Simulator state (Step {step})")

    ani = FuncAnimation(fig, update, frames=len(Estimator_list), interval=500)
    
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.startswith(base_filename) and f.endswith(".gif")]
    numbers = [int(f.split("_")[-1].split(".")[0]) for f in existing if "_" in f and f.split("_")[-1].split(".")[0].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    filename = os.path.join(folder, f"{base_filename}_{next_number}.gif")

    ani.save(filename, writer="ffmpeg", fps=2)
    print(f"Animation saved to: {filename}")



def vector_posterior_animation(Estimator, folder="animations", base_filename="evolution_plot"):
    
    num_steps = len(Estimator.history_list)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    line_prior, = ax.plot([], [], linestyle="-", color="blue", label="Prior")
    line_posterior, = ax.plot([], [], linestyle="-.", color="green", label="Posterior")

    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title("Evolution of Prior, Likelihood, and Posterior")
    ax.legend()
    ax.grid(True)

    
    xlim = len(Estimator.history_list[0]["prior"])
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 1)  

    def update(step):
        entry = Estimator.history_list[step]
        prior = np.array(entry["prior"])
        posterior = np.array(entry["posterior"])

        line_prior.set_data(range(len(prior)), prior)
        line_posterior.set_data(range(len(posterior)), posterior)
        ax.set_title(f"Evolution of Prior, Likelihood, and Posterior (Step {step})")
        return line_prior, line_posterior

    ani = FuncAnimation(fig, update, frames=num_steps, interval=200, blit=False)

    
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.startswith(base_filename) and f.endswith(".gif")]
    numbers = [int(f.split("_")[-1].split(".")[0]) for f in existing if "_" in f and f.split("_")[-1].split(".")[0].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    filename = os.path.join(folder, f"{base_filename}_{next_number}.gif")

    ani.save(filename, writer="ffmpeg", fps=2)
    print(f"Animation saved to: {filename}")





def plot_cross_entropies_vs_step(curves_by_label, color_map, filename = "figure.png"):

    fig, ax = plt.subplots(figsize=(8, 4))
    seen_labels = set()

    for label, curves in curves_by_label.items():
        for steps, ce in curves:
            ax.plot(steps, ce, linestyle='-', color=color_map[label], alpha=0.1,
                    label=label if label not in seen_labels else "")
            seen_labels.add(label)

    ax.set_xlabel("Step", fontsize=20)
    ax.set_ylabel("Cross-Entropy", fontsize=20)
    ax.set_title("Monte Carlo Initialization: Cross-Entropy vs Step", fontsize=25)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad = 8)


    fig.tight_layout()

    plot_bayesian_run(fig, filename)

    plt.show()


    name, ext = os.path.splitext(filename)
    filename_category = f"{name}_per_category{ext}"


    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    label_order = ['LM', 'PU', 'PL', 'Other']

    for ax, label in zip(axs.flat, label_order):
        for steps, ce in curves_by_label[label]:
            ax.plot(steps, ce, linestyle='-', color=color_map[label], alpha=0.2)
        ax.set_title(f"{label}")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("Step", fontsize=20)
        ax.set_ylabel("Cross-Entropy", fontsize=20)
        ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad = 8)


    fig.suptitle("Cross-Entropy vs Step per Category", fontsize=25)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plot_bayesian_run(fig, filename_category)

    plt.show()


    total_elements = sum(len(lst) for lst in curves_by_label.values())
    print("Total number of initializations:", total_elements)

    for label, lst in curves_by_label.items():
        print(f"{label}: {len(lst)} Initializations")


        




def histo_final_cross_entropy(curves_by_label, color_map, only_total = False, filename = "figure_histo.png"):

    label_order = ['LM', 'PU', 'PL', 'Other']

    # Step 1: raccogli tutti i last_values positivi per trovare min e max globali
    all_last_values = []
    curves_last_values = {}


    for label in label_order:
        last_values = [ce[-1] for _, ce in curves_by_label[label] if len(ce) > 0]
        last_values = np.array(last_values)
        negativi = last_values[last_values <= 0]
        if len(negativi) > 0:
            print(f"Valori negativi o zero per {label}: {negativi}")
        # last_values = last_values[last_values > 0]
        all_last_values.extend(last_values)
        curves_last_values[label] = last_values

    global_min = min(all_last_values)
    global_max = max(all_last_values)

    # Step 2: definisci i bin logaritmici comuni
    bins = np.logspace(np.log10(global_min), np.log10(global_max), 100)
    vline = -np.log(0.9)

    if not only_total:
        fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True)

        name, ext = os.path.splitext(filename)
        filename_category = f"{name}_per_category{ext}"

        # Step 3: plottaggio con asse secondario
        for ax, label in zip(axs.flat, label_order):
            last_values = curves_last_values[label]

            ax.hist(last_values, bins=bins, color=color_map[label], alpha=0.7, edgecolor='black')
            ax.set_xscale('log')
            ax.axvline(vline, color='red', linestyle='--', label='H @ p=0.9')
            ax.set_title(f"{label} – Final cross-entropy", fontsize=25)
            ax.set_ylabel("Count", fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.5, which='both')
            ax.legend(fontsize=20)
            ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad = 8)


        # Etichette globali
        for ax in axs[0]:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
            ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad = 8)


        fig.supxlabel("Cross-entropy", fontsize=20)
        fig.suptitle("Histogram of final Cross-entropy values per category", fontsize=25)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plot_bayesian_run(fig, filename_category)

        plt.show()


    fig, ax = plt.subplots(figsize=(8, 4))

    for label in label_order:
        last_values = curves_last_values[label]
        ax.hist(last_values, bins=bins, alpha=0.6, label=label,
                color=color_map[label], edgecolor='black')

    ax.axvline(vline, color='red', linestyle='--', label='H @ p=0.9')
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.5, which='both')
    ax.set_xlabel("Cross-entropy", fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.set_title("Combined histogram of final cross-entropy values by category", fontsize=25)
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', labelcolor="black", labelsize=20, pad = 8)


    fig.tight_layout()

    plot_bayesian_run(fig, filename)

    plt.show()


    count_less = 0
    count_greater = 0

    for values in curves_last_values.values():
        count_less += np.sum(values < vline)
        count_greater += np.sum(values >= vline)

    print(f"Number of success events (H < -log(0.9) ): {count_less}")
    print(f"Number of failure events (H >= -log(0.9) ): {count_greater}", '\n')

    total_runs = sum(len(v) for v in curves_last_values.values())
    success_rate_total = count_less / total_runs

    leftmost_values = curves_last_values['LM']
    leftmost_successes = np.sum(leftmost_values < vline)
    leftmost_total = len(leftmost_values)
    success_rate_leftmost = leftmost_successes / leftmost_total if leftmost_total > 0 else np.nan

    print(f"Total initializations: {total_runs}")
    print(f"Total success probability: {success_rate_total:.3f}")
    print(f"Leftmost initializations: {leftmost_total}")
    print(f"Success probability given Leftmost: {success_rate_leftmost:.3f}")

    return total_runs, success_rate_total, leftmost_total, success_rate_leftmost





# Funzione interna per plot con controllo limiti
def plot_with_clipped_errors(ax, x, mean, std, median, title, N, num_methods, colors, method_names, success_rate_total=None):
    for i in range(num_methods):
        lower = max(0, mean[i] - std[i])


        if title == "Probability":
            upper = min(1, mean[i] + std[i])
        else:  # "Steps"
            upper = mean[i] + std[i]

        yerr = np.array([[mean[i] - lower], [upper - mean[i]]])

        if success_rate_total is not None:
            std_succ = np.sqrt(success_rate_total[i] * (1 - success_rate_total[i]) / N) 
            lower_succ = max(0, success_rate_total[i] - std_succ)
            upper_succ = min(1, success_rate_total[i] + std_succ)

            yerr_succ = np.array([[success_rate_total[i] - lower_succ], [upper_succ - success_rate_total[i]]])

        # Quadrato per la media con barre di errore
        ax.errorbar(x[i], mean[i], yerr=yerr, fmt='s',  # 's' = square
                    color=colors[i], capsize=14, markersize=8, elinewidth=1.5, capthick=2, markeredgecolor='black')

        # Triangolo per la mediana
        ax.plot(x[i], median[i], marker='^',  # '^' = triangle up
                color=colors[i], markersize=5, linestyle='None', markeredgecolor='black')
        
        # Barra per la probabilità di successo
        # ax.axhline(x[i], success_rate_total, color='blue', linestyle='--', label="Probabilità di successo")

        if success_rate_total is not None:
            # ax.plot(x[i], success_rate_total[i], marker='o',  # '^' = triangle up
            #         color=colors[i], markersize=5, linestyle='None', markeredgecolor='black')
            ax.errorbar(x[i], success_rate_total[i], yerr=yerr_succ, fmt='o',  # 's' = square
                    color=colors[i], capsize=8, markersize=5, elinewidth=1, markeredgecolor='black')
            cross_patch = plt.Line2D([0], [0], marker='o', color='black', markersize=5, linestyle='None', label="Success Rate")

        # Barra per la probabilità condizionata
        # ax.axhline(success_rate_total, color='green', linestyle='--', label="Probabilità condizionata al leftmost")

    ax.set_title(title)
    ax.set_xticks(x)
    

    ax.set_xticklabels(method_names, rotation=45)
    # ax.set_yscale('log')
    ax.grid(True, linestyle="--", alpha=0.5)
    if title == "Probability":
        ax.axhline(0.9, color='red', linestyle='--', label='p=0.9')

    triangle_patch = plt.Line2D([0], [0], marker='^', color='black', markersize=5, linestyle='None', label="Median")

    square_patch = plt.Line2D([0], [0], marker='s', color='black', markersize=8,
                          linestyle='None', label="Mean")

    # if success_rate_total is not None:
    #     ax.legend(handles=[triangle_patch, cross_patch] + ax.get_legend_handles_labels()[0])
    # else:
    #     ax.legend(handles=[triangle_patch] + ax.get_legend_handles_labels()[0])



    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)

    # PATCHES per legenda
    square_patch = plt.Line2D([0], [0], marker='s', color='black', markersize=8,
                              linestyle='None', label="Mean")            # ⬛ Media
    triangle_patch = plt.Line2D([0], [0], marker='^', color='black', markersize=6,
                                linestyle='None', label="Median")       # 🔺 Mediana

    handles = [square_patch, triangle_patch]

    if success_rate_total is not None:
        cross_patch = plt.Line2D([0], [0], marker='o', color='black', markersize=6,
                                 linestyle='None', label="Success Rate")  # ⚫ Success rate
        handles.append(cross_patch)

    # Aggiungi eventuali altri handle già generati da errorbar
    existing_handles = ax.get_legend_handles_labels()[0]

    ax.legend(handles=handles + existing_handles, fontsize=10)



def plot_method_comparison(stats_list, method_names, success_rate_total, N, filename="method_comparison.svg"):
    num_methods = len(stats_list)
    x = np.arange(num_methods)

    # Estrai i dati
    prob_mean_success = [s["probability"]["success"]["mean"] for s in stats_list]
    prob_std_success = [s["probability"]["success"]["std"] for s in stats_list]
    prob_median_success = [s["probability"]["success"]["median"] for s in stats_list]

    prob_mean_all = [s["probability"]["all"]["mean"] for s in stats_list]
    prob_std_all = [s["probability"]["all"]["std"] for s in stats_list]
    prob_median_all = [s["probability"]["all"]["median"] for s in stats_list]


    steps_mean_success = [s["steps"]["success"]["mean"] for s in stats_list]
    steps_std_success = [s["steps"]["success"]["std"] for s in stats_list]
    steps_median_success = [s["steps"]["success"]["median"] for s in stats_list]

    steps_mean_all = [s["steps"]["all"]["mean"] for s in stats_list]
    steps_std_all = [s["steps"]["all"]["std"] for s in stats_list]
    steps_median_all = [s["steps"]["all"]["median"] for s in stats_list]

    # Colori professionali
    colors = plt.cm.tab10.colors[:num_methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("Method comparison \n Distribution of the final probability of correct estimation and number of steps", fontsize=14)

    # Subplot: Probabilità di successo
    plot_with_clipped_errors(axes[0], x, prob_mean_all, prob_std_all, prob_median_all, "Probability", N, num_methods, colors, method_names, success_rate_total=success_rate_total)
    # axes[0].legend()
    axes[0].set_ylabel("Probability")

    # Subplot: Probabilità totale
    plot_with_clipped_errors(axes[1], x, steps_mean_all, steps_std_all, steps_median_all, "Steps", N, num_methods, colors, method_names)
    # axes[1].legend()
    axes[1].set_ylabel("Steps")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure_in_images(fig, filename)

    plt.show()