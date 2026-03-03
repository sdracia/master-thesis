from pathlib import Path
import re
import joblib
from QLS.state_dist import States
import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, basis, qeye, expect

from saving import save_figure_in_images



# ==========================================================
# FUNZIONE UTILITY: cerca cartella risalendo il tree
# ==========================================================
def find_data_dir(folder_name, start_path=Path.cwd()):
    """
    Cerca folder_name risalendo il filesystem dal percorso start_path.
    Restituisce Path se trovato, altrimenti solleva FileNotFoundError.
    """
    current = start_path.resolve()

    while True:
        candidate = current / folder_name
        if candidate.exists() and candidate.is_dir():
            return candidate

        if current.parent == current:
            # siamo arrivati alla radice senza trovare la cartella
            raise FileNotFoundError(
                f"Directory '{folder_name}' non trovata risalendo da {start_path}"
            )
        current = current.parent


# ==========================================================
# COMMON LOADER
# ==========================================================
def _load_common(molecule, temperature, data_dir):
    """
    Parte comune a CaH e CaOH:
    - costruzione distribuzione termica
    - caricamento file Jxx.pkl ordinati
    """
    states = States(molecule, temperature)
    j_dist = states.j_distribution()

    pattern = re.compile(r"J(\d+)_")

    data_files = sorted(
        [f for f in data_dir.glob("*.pkl") if pattern.match(f.name)],
        key=lambda f: int(pattern.match(f.name).group(1))
    )

    return states, j_dist, data_files


# ==========================================================
# CaH PIPELINE
# ==========================================================
def _run_cah(molecule, temperature, j_max, plot=False, rap_flag = True, savetext = "cah_after_pumping_rap_pop.svg"):

    # --- Trova la cartella pumping_RAP_data_cah risalendo il tree ---
    data_dir = find_data_dir("pumping_RAP_data_cah")

    states, j_dist, data_files = _load_common(
        molecule, temperature, data_dir
    )

    # --- Pumping manifolds fit ---
    pop_fit_0, pop_fit_1, pop_fit_2jp1 = pumping_manifolds_cah(
        data_files, plot=plot
    )

    # --- RAP signature ---
    data_cah = joblib.load(data_dir / "rap_signature_cah.pkl")
    pop_vals_cah = data_cah["pop_vals"]

    # --- After pumping populations ---
    (
        pop0_before, pop1_before, pop2jp1_before,
        pop0_after, pop1_after, pop2jp1_after
    ) = after_pumping_rap_pop_cah(
        molecule=molecule,
        state_dist=states,
        j_max=j_max,
        pop_fit_0=pop_fit_0,
        pop_fit_1=pop_fit_1,
        pop_fit_2jp1=pop_fit_2jp1,
        rap_sign_cah=pop_vals_cah,
        plot=plot,
        rap_flag=rap_flag,
        savetext=savetext
    )

    pop_fit = [pop_fit_0, pop_fit_1, pop_fit_2jp1]

    return {
        "states": states,
        "j_dist": j_dist,
        "pop_fit": pop_fit,
        "before": (pop0_before, pop1_before, pop2jp1_before),
        "after": (pop0_after, pop1_after, pop2jp1_after),
    }


# ==========================================================
# CaOH PIPELINE
# ==========================================================
def _run_caoh(molecule, temperature, j_max, plot=False, rap_flag = True, savetext = "caoh_after_pumping_rap_pop.svg"):

    # --- Trova la cartella pumping_RAP_data_caoh risalendo il tree ---
    data_dir = find_data_dir("pumping_RAP_data_caoh")

    states, j_dist, data_files = _load_common(
        molecule, temperature, data_dir
    )

    # --- Pumping manifolds fit ---
    pop_fit_0, pop_fit_1, pop_fit_2jp1 = pumping_manifolds(
        data_files, plot=plot
    )

    # --- RAP signatures ---
    data_high = joblib.load(data_dir / "rap_signature_high.pkl")
    data_middle = joblib.load(data_dir / "rap_signature_middle.pkl")
    data_low = joblib.load(data_dir / "rap_signature_low.pkl")
    data_LL = joblib.load(data_dir / "rap_signature_LL.pkl")

    pop_vals_high = data_high["pop_vals"]
    pop_vals_middle = data_middle["pop_vals"]
    pop_vals_low = data_low["pop_vals"]
    pop_vals_LL = data_LL["pop_vals"]

    # --- After pumping populations ---
    (
        pop0_before, pop1_before, pop2jp1_before,
        pop0_after, pop1_after, pop2jp1_after
    ) = after_pumping_rap_pop(
        molecule=molecule,
        state_dist=states,
        j_max=j_max,
        pop_fit_0=pop_fit_0,
        pop_fit_1=pop_fit_1,
        pop_fit_2jp1=pop_fit_2jp1,
        rap_sign_low=pop_vals_low,
        rap_sign_middle=pop_vals_middle,
        rap_sign_high=pop_vals_high,
        rap_sign_LL=pop_vals_LL,
        plot=plot,
        rap_flag=rap_flag,
        savetext=savetext
    )


    pop_fit = [pop_fit_0, pop_fit_1, pop_fit_2jp1]

    return {
        "states": states,
        "j_dist": j_dist,
        "pop_fit": pop_fit,
        "before": (pop0_before, pop1_before, pop2jp1_before),
        "after": (pop0_after, pop1_after, pop2jp1_after),
    }


# ==========================================================
# WRAPPER PRINCIPALE
# ==========================================================
def run_pumping_pipeline(
    molecule_type,
    molecule,
    temperature,
    j_max,
    plot=False,
    rap_flag = True, 
):
    """
    Wrapper unico per CaH e CaOH.
    Cerca automaticamente le cartelle corrette nel filesystem.
    """

    molecule_type = molecule_type.lower()

    if molecule_type == "cah":
        return _run_cah(molecule, temperature, j_max, plot, rap_flag, savetext="cah_after_pumping_rap_pop.svg")

    elif molecule_type == "caoh":
        return _run_caoh(molecule, temperature, j_max, plot, rap_flag, savetext="caoh_after_pumping_rap_pop.svg")

    else:
        raise ValueError(
            f"Unknown molecule_type '{molecule_type}'. "
            "Supported: 'CaH', 'CaOH'."
        )
    


# results = run_pumping_pipeline(
#     molecule_type="CaOH",
#     molecule=molecule,
#     temperature=temperature,
#     j_max=j_max,
#     plot=True
# )

# # Popolazioni prima del RAP
# pop0_before, pop1_before, pop2jp1_before = results["before"]

# # Popolazioni dopo il RAP
# pop0_after, pop1_after, pop2jp1_after = results["after"]

# # Fit dei valori di pumping
# pop_fit_0, pop_fit_1, pop_fit_2jp1 = results["pop_fit"]







#################################
##### PUMPING * RAP SCHEME ######
#################################

def fit_populations(popj_dict, type, no_plot = True, ax = None):
    # j_vals = np.array([1, 3, 5, 8])
    # pop_vals = np.array([0.5, population_j3[0], population_j5[0], population_j8[0]])

    j_vals = np.array(sorted(popj_dict.keys()))
    pop_vals = np.array([popj_dict[j] for j in j_vals])

    def sigmoid_1(J, A, B, C):
        # return 0.5 + 0.5 / (1 + A * np.exp(-B * (J - C)))
        return 0.5 / (1 + A * np.exp(-B * (J - C)))
    
    def sigmoid_2jp1(J, A, B, C):
        # return 0.5 + 0.5 / (1 + A * np.exp(-B * (J - C)))
        return 0.5 - 0.5 / (1 + A * np.exp(-B * (J - C)))


    # Fit dei parametri
    if type == "popj_1":
        popt, _ = curve_fit(sigmoid_1, j_vals, pop_vals)
    elif type == "popj_2jp1":
        popt, _ = curve_fit(sigmoid_2jp1, j_vals, pop_vals)
    else:
        print("Invalid type. Choose 'popj_1' or 'popj_2jp1'.")


    A_fit, B_fit, C_fit = popt

    # Valori interpolati da j=1 a j=50
    j_fit = np.arange(1, 51)

    if type == "popj_1":
        pop_fit = sigmoid_1(j_fit, A_fit, B_fit, C_fit)
        color = '#4B0082'
        label = "PU state"
    elif type == "popj_2jp1":
        pop_fit = sigmoid_2jp1(j_fit, A_fit, B_fit, C_fit)
        color = '#DAA520'
        label = "PL state"
    else:
        print("Invalid type. Choose 'popj_1' or 'popj_2jp1'.")
    
    

    if not no_plot:
        # Plot
        ax.scatter(j_vals, pop_vals, label=label, color=color)
        ax.plot(j_fit, pop_fit, label="fit " + label, color=color, linestyle='-')

    return pop_fit


def pumping_manifolds(data_files, plot = True, savetext = "caoh_after_pumping.svg"):
    popj_0_dict = {}
    popj_1_dict = {}
    popj_2jp1_dict = {}


    for file_path in data_files:
        data = joblib.load(file_path)

        j_val = data["j_val"]
        rho_final = data["rho_final"]
        args = data["args"]
        n_motional = args["terms"][0]["n_motional"]
        n_internal = args["terms"][0]["n_internal"]

        populations = np.zeros(n_internal)
                                
        for j in range(n_internal):
            Pj_op = tensor(basis(n_internal, j) * basis(n_internal, j).dag(), qeye(n_motional))
            populations[j] = expect(Pj_op, rho_final)

        popj_0 = populations[0]
        popj_1 = populations[1]

        if j_val >=11 :
            popj_0 = popj_0 - 0.1 + 1/(2*(2*j_val + 1)) 
            popj_1 = popj_1 + 0.1 - 1/(2*(2*j_val + 1)) 


        index = min(2*j_val + 1, int(n_internal/2))
        popj_2jp1 = populations[index]

        popj_0_dict[j_val] = popj_0
        popj_1_dict[j_val] = popj_1
        popj_2jp1_dict[j_val] = popj_2jp1

    if plot==True:
        fig, ax = plt.subplots(figsize=(10, 5))

        pop_fit_1 = fit_populations(popj_1_dict, "popj_1", no_plot=False, ax = ax)
        pop_fit_2jp1 = fit_populations(popj_2jp1_dict, "popj_2jp1", no_plot=False, ax = ax)
        pop_fit_0 = 1 - pop_fit_1 - pop_fit_2jp1

        j_vals_0 = np.array(sorted(popj_0_dict.keys()))
        pop_vals_0 = np.array([popj_0_dict[j] for j in j_vals_0])
        j_fit = np.arange(1, 51)

        ax.scatter(j_vals_0, pop_vals_0, label="LM (target) state", color='#800020')
        ax.plot(j_fit, pop_fit_0, label="fit LM state", color='#800020', linestyle='-')

        ax.set_xlabel(r'Manifold $J$', fontsize=25)
        ax.set_ylabel('Population', fontsize=25)
        ax.set_title('Population after pumping stage', fontsize=28)
        ax.grid(True)
        ax.legend(fontsize=21, frameon=True)

        ax.tick_params(axis='both', which='major', labelsize=25)

        fig.tight_layout()
        ax.set_ylim(-0.05, 1.05)

        save_figure_in_images(fig, savetext)

        plt.show()

    else:
        pop_fit_1 = fit_populations(popj_1_dict, "popj_1", no_plot = True)
        pop_fit_2jp1 = fit_populations(popj_2jp1_dict, "popj_2jp1", no_plot = True)
        pop_fit_0 = 1 - pop_fit_1 - pop_fit_2jp1


        j_vals_0 = np.array(sorted(popj_0_dict.keys()))
        pop_vals_0 = np.array([popj_0_dict[j] for j in j_vals_0])
        j_fit = np.arange(1, 51)


    return pop_fit_0, pop_fit_1, pop_fit_2jp1


def after_pumping_rap_pop(molecule, state_dist, j_max, 
                          pop_fit_0, pop_fit_1, pop_fit_2jp1,
                          rap_sign_low, rap_sign_middle, rap_sign_high, rap_sign_LL, plot=True, rap_flag = True,
                          savetext = "caoh_after_pumping_rap_pop.svg"):
    
    thermal_distribution = state_dist.j_distribution()

    # Reset of column state_dist, from J=1 on. J=0 keeps the thermal distribution since no coupling is envolved. 
    molecule.state_df.loc[molecule.state_df.index[2:], "state_dist"] = 0.0

    total_check = pop_fit_0 + pop_fit_1 + pop_fit_2jp1
    if not np.allclose(total_check, 1.0, atol=1e-6):
        raise ValueError(f"Population is not normalized to 1")
    

    modified_pop0 = []
    modified_pop1 = []
    modified_pop2 = []


    for j_val in range(1, j_max + 1):
        tot_pop_in_j = thermal_distribution[j_val]

        # Pumping
        pop0 = pop_fit_0[j_val - 1] * tot_pop_in_j
        pop1 = pop_fit_1[j_val - 1] * tot_pop_in_j
        pop2 = pop_fit_2jp1[j_val - 1] * tot_pop_in_j

        if rap_flag:
            # First RAP
            transfer = rap_sign_low[j_val] * pop1
            pop0 = pop0 + transfer
            pop1 = pop1 - transfer

            # Second RAP
            transfer = rap_sign_middle[j_val] * pop1
            pop0 = pop0 + transfer
            pop1 = pop1 - transfer

            # Third RAP
            transfer = rap_sign_high[j_val] * pop1
            pop0 = pop0 + transfer
            pop1 = pop1 - transfer

            # Fourth RAP
            transfer = rap_sign_LL[j_val] * pop2
            pop0 = pop0 + transfer
            pop2 = pop2 - transfer


        modified_pop0.append(pop0/tot_pop_in_j)
        modified_pop1.append(pop1/tot_pop_in_j)
        modified_pop2.append(pop2/tot_pop_in_j)


        states_in_j = molecule.state_df.loc[molecule.state_df["j"] == j_val].copy()
        m_len = 2 * j_val + 1
        num_states = 2 * m_len

        # Obtain the real indices in the original DataFrame 
        indices_in_j = molecule.state_df.index[molecule.state_df["j"] == j_val]

        molecule.state_df.loc[indices_in_j[0], "state_dist"] = pop0
        molecule.state_df.loc[indices_in_j[1], "state_dist"] = pop1
        molecule.state_df.loc[indices_in_j[m_len], "state_dist"] = pop2

        # Residual population in the background is added (imperfect pumping): 5% of the total population in the j sublevel
        population_off = 0.05 * tot_pop_in_j / num_states
        molecule.state_df.loc[indices_in_j, "state_dist"] += population_off

        # Renormalization of the populations in the j manifold
        total_in_j = molecule.state_df.loc[indices_in_j, "state_dist"].sum()
        molecule.state_df.loc[indices_in_j, "state_dist"] = tot_pop_in_j * molecule.state_df.loc[indices_in_j, "state_dist"] / total_in_j

    
    modified_pop0 = np.array(modified_pop0)
    modified_pop1 = np.array(modified_pop1)
    modified_pop2 = np.array(modified_pop2)

    j_fit = np.arange(1, 51)

    if plot==True:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(j_fit, modified_pop0, label="LM (target) state", color='#800020', linestyle='-')
        ax.plot(j_fit, modified_pop1, label="PU state", color='#4B0082', linestyle='-')
        ax.plot(j_fit, modified_pop2, label="PL state", color='#DAA520', linestyle='-')

        ax.set_xlabel(r'Manifold $J$', fontsize=25)
        ax.set_ylabel('Population', fontsize=25)
        ax.set_title('Population after pumping and RAP stages', fontsize=28)

        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.legend(fontsize=21, frameon=True, loc = "center left")

        ax.tick_params(axis='both', which='major', labelsize=25)
        fig.tight_layout()

        save_figure_in_images(fig, savetext)

        plt.show()




    return pop_fit_0, pop_fit_1, pop_fit_2jp1, modified_pop0, modified_pop1, modified_pop2

    

from scipy.optimize import curve_fit


def fit_populations_cah(popj_dict, type, no_plot = True, ax = None):
    # j_vals = np.array([1, 3, 5, 8])
    # pop_vals = np.array([0.5, population_j3[0], population_j5[0], population_j8[0]])

    j_vals = np.array(sorted(popj_dict.keys()))
    pop_vals = np.array([popj_dict[j] for j in j_vals])

    def sigmoid_1_cah(J, A, B, C):
        # return 0.5 + 0.5 / (1 + A * np.exp(-B * (J - C)))
        return 0.475 / (1 + A * np.exp(-B * (J - C)))
    
    def sigmoid_2jp1_cah(J, A, B, C):
        # return 0.5 + 0.5 / (1 + A * np.exp(-B * (J - C)))
        return 0.5 - 0.5 / (1 + A * np.exp(-B * (J - C)))


    # Fit dei parametri
    if type == "popj_1":
        p0 = [0.7, 1.3, 5.7]
        popt, _ = curve_fit(sigmoid_1_cah, j_vals, pop_vals, maxfev=5000, p0=p0)
    elif type == "popj_2jp1":
        p0 = [0.1, 1.8, 1]
        popt, _ = curve_fit(sigmoid_2jp1_cah, j_vals, pop_vals, maxfev=10000, p0=p0)
    else:
        print("Invalid type. Choose 'popj_1' or 'popj_2jp1'.")


    A_fit, B_fit, C_fit = popt

    # Valori interpolati da j=1 a j=50
    j_fit = np.arange(1, 15, 0.25)

    if type == "popj_1":
        pop_fit = sigmoid_1_cah(j_fit, A_fit, B_fit, C_fit)
        color = '#4B0082'
        label = "PU state"
    elif type == "popj_2jp1":
        pop_fit = sigmoid_2jp1_cah(j_fit, A_fit, B_fit, C_fit)
        color = '#DAA520'
        label = "PL state"
    else:
        print("Invalid type. Choose 'popj_1' or 'popj_2jp1'.")
    
    

    if not no_plot:
        # Plot
        ax.scatter(j_vals, pop_vals, label=label, color=color)
        ax.plot(j_fit, pop_fit, label="fit " + label, color=color, linestyle='-')

    return pop_fit


def pumping_manifolds_cah(data_files, plot = True, savetext = "cah_after_pumping.svg"):
    popj_0_dict = {}
    popj_1_dict = {}
    popj_2jp1_dict = {}


    for file_path in data_files:
        data = joblib.load(file_path)

        j_val = data["j_val"]
        rho_final = data["rho_final"]
        args = data["args"]
        n_motional = args["terms"][0]["n_motional"]
        n_internal = args["terms"][0]["n_internal"]

        populations = np.zeros(n_internal)
                                
        for j in range(n_internal):
            Pj_op = tensor(basis(n_internal, j) * basis(n_internal, j).dag(), qeye(n_motional))
            populations[j] = expect(Pj_op, rho_final)

        popj_0 = populations[0]
        popj_1 = populations[1]

        if j_val >=8 :
            popj_0 = popj_0 - 0.1 + 1/(2*(2*j_val + 1)) 
            popj_1 = popj_1 + 0.1 - 1/(2*(2*j_val + 1)) 


        index = min(2*j_val + 1, int(n_internal/2))
        popj_2jp1 = populations[index]

        popj_0_dict[j_val] = popj_0
        popj_1_dict[j_val] = popj_1
        popj_2jp1_dict[j_val] = popj_2jp1

    if plot==True:
        fig, ax = plt.subplots(figsize=(10, 5))

        pop_fit_1 = fit_populations_cah(popj_1_dict, "popj_1", no_plot=False, ax = ax)
        pop_fit_2jp1 = fit_populations_cah(popj_2jp1_dict, "popj_2jp1", no_plot=False, ax = ax)

        pop_fit_0 = 1 - pop_fit_1 - pop_fit_2jp1

        j_vals_0 = np.array(sorted(popj_0_dict.keys()))
        pop_vals_0 = np.array([popj_0_dict[j] for j in j_vals_0])
        j_fit = np.arange(1, 15,0.25)

        ax.scatter(j_vals_0, pop_vals_0, label="LM (target) state", color='#800020')
        ax.plot(j_fit, pop_fit_0, label="fit LM state", color='#800020', linestyle='-')

        ax.set_xlabel(r'Manifold $J$', fontsize=25)
        ax.set_ylabel('Population', fontsize=25)
        ax.set_title('Population after pumping stage', fontsize=28)
        ax.grid(True)
        ax.legend(fontsize=21, loc = "upper right", frameon=True)

        ax.tick_params(axis='both', which='major', labelsize=25)

        fig.tight_layout()
        ax.set_ylim(-0.05, 1.05)

        save_figure_in_images(fig, savetext)

        plt.show()

    else:
        pop_fit_1 = fit_populations_cah(popj_1_dict, "popj_1", no_plot = True)
        pop_fit_2jp1 = fit_populations_cah(popj_2jp1_dict, "popj_2jp1", no_plot = True)
        pop_fit_0 = 1 - pop_fit_1 - pop_fit_2jp1


        j_vals_0 = np.array(sorted(popj_0_dict.keys()))
        pop_vals_0 = np.array([popj_0_dict[j] for j in j_vals_0])
        j_fit = np.arange(1, 15)


    indices_integers = np.arange(0, len(j_fit), 4)
    pop_fit_0 = pop_fit_0[indices_integers]
    pop_fit_1 = pop_fit_1[indices_integers]
    pop_fit_2jp1 = pop_fit_2jp1[indices_integers]

    return pop_fit_0, pop_fit_1, pop_fit_2jp1




def after_pumping_rap_pop_cah(molecule, state_dist, j_max, 
                          pop_fit_0, pop_fit_1, pop_fit_2jp1,
                          rap_sign_cah, plot=True, rap_flag = True,
                          savetext = "cah_after_pumping_rap_pop.svg"):
    
    thermal_distribution = state_dist.j_distribution()

    # Reset of column state_dist, from J=1 on. J=0 keeps the thermal distribution since no coupling is envolved. 
    molecule.state_df.loc[molecule.state_df.index[2:], "state_dist"] = 0.0

    total_check = pop_fit_0 + pop_fit_1 + pop_fit_2jp1
    if not np.allclose(total_check, 1.0, atol=1e-6):
        raise ValueError(f"Population is not normalized to 1")
    

    modified_pop0 = []
    modified_pop1 = []
    modified_pop2 = []


    for j_val in range(1, j_max + 1):
        tot_pop_in_j = thermal_distribution[j_val]

        # Pumping
        pop0 = pop_fit_0[j_val - 1] * tot_pop_in_j
        pop1 = pop_fit_1[j_val - 1] * tot_pop_in_j
        pop2 = pop_fit_2jp1[j_val - 1] * tot_pop_in_j

        if rap_flag:
            # First RAP
            transfer = rap_sign_cah[j_val] * pop1
            pop0 = pop0 + transfer
            pop1 = pop1 - transfer



        modified_pop0.append(pop0/tot_pop_in_j)
        modified_pop1.append(pop1/tot_pop_in_j)
        modified_pop2.append(pop2/tot_pop_in_j)


        states_in_j = molecule.state_df.loc[molecule.state_df["j"] == j_val].copy()
        m_len = 2 * j_val + 1
        num_states = 2 * m_len

        # Obtain the real indices in the original DataFrame 
        indices_in_j = molecule.state_df.index[molecule.state_df["j"] == j_val]

        molecule.state_df.loc[indices_in_j[0], "state_dist"] = pop0
        molecule.state_df.loc[indices_in_j[1], "state_dist"] = pop1
        molecule.state_df.loc[indices_in_j[m_len], "state_dist"] = pop2

        # Residual population in the background is added (imperfect pumping): 5% of the total population in the j sublevel
        population_off = 0.05 * tot_pop_in_j / num_states
        molecule.state_df.loc[indices_in_j, "state_dist"] += population_off

        # Renormalization of the populations in the j manifold
        total_in_j = molecule.state_df.loc[indices_in_j, "state_dist"].sum()
        molecule.state_df.loc[indices_in_j, "state_dist"] = tot_pop_in_j * molecule.state_df.loc[indices_in_j, "state_dist"] / total_in_j

    
    modified_pop0 = np.array(modified_pop0)
    modified_pop1 = np.array(modified_pop1)
    modified_pop2 = np.array(modified_pop2)

    j_fit = np.arange(1, j_max + 1)

    if plot==True:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(j_fit, modified_pop0, label="LM (target) state", color='#800020', linestyle='-')
        ax.plot(j_fit, modified_pop1, label="PU state", color='#4B0082', linestyle='-')
        ax.plot(j_fit, modified_pop2, label="PL state", color='#DAA520', linestyle='-')

        ax.set_xlabel(r'Manifold $J$', fontsize=25)
        ax.set_ylabel('Population', fontsize=25)
        ax.set_title('Population after pumping and RAP stages', fontsize=28)

        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.legend(fontsize=21, frameon=True, loc = "center left")

        ax.tick_params(axis='both', which='major', labelsize=25)
        fig.tight_layout()

        save_figure_in_images(fig, savetext)

        plt.show()




    return pop_fit_0, pop_fit_1, pop_fit_2jp1, modified_pop0, modified_pop1, modified_pop2
