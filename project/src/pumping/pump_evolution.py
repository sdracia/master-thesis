from qutip import *

from pumping.pump_utils import *
from saving import save_final_state
from pumping.pump_plotting import plot_internal_and_motional_dynamics
from pumping.pump_hamiltonians import H_bsb_manifold
from pumping.pump_operators import collapse_cooling_op



def run_pumping (b_field_gauss, j_max, temperature, j_val, n_motional, 
                 laser_configs, e_ops,
                 molecule_type = "CaH", 
                 save_intermediate_states=False, plot_intermediate_dynamics=False):
    
    validate_laser_configs(laser_configs)


    mo1, states1, mo1_dm2, states1_dm2 = initialize_molecule(
        molecule_type,
        b_field_gauss,
        j_max,
        temperature
    )

    transitions_in_j = mo1.transition_df[mo1.transition_df["j"] == j_val]
    transitions_in_j_dm2 = mo1_dm2.transition_df[mo1_dm2.transition_df["j"] == j_val]

    n_internal = 2*(2 * j_val + 1)        # if the entire manifold is considered, the number of internal states is 2*(2J+1) 



    if j_val >8 :
        keep_sub_manifold_levels = 5
        transitions_in_j = cut_trans_df(transitions_in_j, j_val, keep_sub_manifold_levels)
        transitions_in_j_dm2 = cut_trans_df(transitions_in_j_dm2, j_val, keep_sub_manifold_levels)

        sub_index = min(keep_sub_manifold_levels, 2*j_val + 1) 
        n_internal = 2*sub_index  

        

    states = [basis(n_internal, i) for i in range(n_internal)]
    rho_internal = sum([ket2dm(state) for state in states]) / n_internal


    # initial motional state
    index_motional = 0
    assert 0 <= index_motional < n_motional, "index_motional out of range"

    psi_motional = basis(n_motional, index_motional)
    rho_motional = ket2dm(psi_motional)

    ## mixed state density matrix
    rho0 = tensor(rho_internal, rho_motional)


    opts = Options(store_states=True, progress_bar="text", nsteps=20000)

    results = []

    scheme_id = "".join([pulse["raman_config"][-1] for pulse in laser_configs])

    for i, config in enumerate(laser_configs):

            is_minus = config["is_minus"]
            times = config["times"]
            laser_detuning = config["laser_detuning"]
            rabi_rate = config["rabi_rate"]
            manifold_type = config["raman_config"]

            if manifold_type == "dm2":
                transitions_selected = transitions_in_j_dm2
            else:
                transitions_selected = transitions_in_j

            cooling_rate = rabi_rate
            final_time = times[-1]

            c_ops = collapse_cooling_op(cooling_rate, n_internal, n_motional)
            H_tot, args = H_bsb_manifold(transitions_selected, j_val, is_minus, n_motional, n_internal, rabi_rate, laser_detuning)

            result = mesolve(H_tot, rho0, times, c_ops, e_ops, args=args, options=opts)

            if save_intermediate_states:
                full_path = save_final_state(result, args, final_time, b_field_gauss, j_val, rabi_rate, laser_detuning, cooling_rate, molecule_type)
                if i == len(laser_configs)-1:
                    full_path = save_final_state(result, args, final_time, b_field_gauss, j_val, rabi_rate, laser_detuning, cooling_rate, molecule_type, last_pulse=True)
                

            if plot_intermediate_dynamics:
                savetext = (
                     f"{molecule_type}_j{j_val}_pulse{i+1}_"
                     f"{len(laser_configs)}stg_{scheme_id}.svg"
                )
                plot_internal_and_motional_dynamics(result, times, n_internal, n_motional, title = "Population evolution", savetext = savetext, only_pop=False)

            rho0 = result.states[-1]

            results.append([result, times])

    savetext = (
        f"{molecule_type}_j{j_val}_"
        f"{len(laser_configs)}stg_{scheme_id}.svg"
    )

    plot_internal_and_motional_dynamics(results, times=None, n_internal=n_internal, n_motional=n_motional, title = "Population evolution", savetext = savetext, only_pop=True)