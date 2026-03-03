from qutip import *
import numpy as np

from ODF.odf_operators import decay, x_op_3mol_atom, p_op_3mol_atom
from ODF.odf_plotting import plot_atom_mol3_evolution, plot_odf_3mol_data
from ODF.odf_hamiltonians import *


###################################################################################
########################## MOLECULE + MOTION + ION ################################
########################## MOLECULE WITH 3 LEVELS  ################################
###################################################################################

def molec3_atom_evolution(psi0, Ham, times, times_atom, det, x, p, N, 
                          rr = None, rr_mol = None, w_mol = None, rr_atom = None, 
                          var_time = True, coherence = False, return_results = False
                          ):

    excitation = []
    spin_down_list = []
    spin_up_list = []
    spin_aux_list = []
    atom_spin_up_list = []
    atom_spin_down_list = []
    results = []

    coherence_time = 100
    if coherence:
        c_ops = [tensor(decay(coherence_time), qeye(N), qeye(2))]
    else:
        c_ops = []

    # opts = Options(atol=1e-10, rtol=1e-10, progress_bar='text', store_states=True)    
    opts = Options(atol=1e-10, rtol=1e-10, store_states=True)    

    # Precompute expectation operators
    n_op = tensor(qeye(3), num(N), qeye(2))
    P_aux = tensor(basis(3, 2) * basis(3, 2).dag(), qeye(N), qeye(2))
    P_down = tensor(basis(3, 1) * basis(3, 1).dag(), qeye(N), qeye(2))
    P_up   = tensor(basis(3, 0) * basis(3, 0).dag(), qeye(N), qeye(2))

    e_ops = [n_op, P_aux, P_down, P_up]


    for delta in det:
         
        # Variable time, with variable loop numbers and fixing a maximum final time
        if var_time:
            time_steps_mol = 50

            t_max = 1.5*1e4     # 15ms
            loop_min = 1
            loop_max = 4

            # To avoid division by 0
            abs_delta = np.abs(delta) if np.abs(delta) > 1e-10 else 1e-10

            # Normalization of |delta| between 0 and 1 wrt the maximum absolute value
            delta_norm = abs_delta / np.max(np.abs(det))

            # Loop number: 1 close to 0, up to 4 for large |delta|
            n_loops = int(np.round(loop_min + (loop_max - loop_min) * delta_norm))

            final_time = min(n_loops * 2 * np.pi / abs_delta, t_max)
            times = np.linspace(0, final_time, time_steps_mol)

            ## Redefinition of the rabi rate of the molecule proportional to the pulse duration
            rr_mol = np.pi / final_time


        args = {}
        if Ham == "bsb":
            args = {'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2 * np.pi * w_mol, 'N': N}
            H = H_mol3_atom_bsb
        elif Ham == "tot":
            args = {'rabi_rate': rr, 'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2 * np.pi * w_mol, 'N': N}
            H = H_mol3_atom_tot
        elif Ham == "odf":
            args = {'rabi_rate': rr, 'd': delta, 'N': N}
            H = H_mol3_atom_odf
        else:
            raise ValueError("Invalid Hamiltonian type")

        # Expectation values only
        result = mesolve(H, psi0, times, c_ops, e_ops, args=args, options=opts)
        final_state_mol = result.states[-1]

        if return_results:
            results.append(result)

        
        excitation.append(result.expect[0][-1])
        spin_aux_list.append([result.expect[1][-1]])
        spin_down_list.append([result.expect[2][-1]])
        spin_up_list.append([result.expect[3][-1]])


        # RBR ATOM READOUT
        args_atom = {'rabi_rate_atom': rr_atom, 'N': N}
        result_atom = mesolve(H_mol3_atom_rsb, final_state_mol, times_atom, [], [], args=args_atom)

        final_state_atom = result_atom.states[-1]

        spin_down_atom = [expect(tensor(qeye(3), qeye(N), basis(2, 1) * basis(2, 1).dag()), final_state_atom)]
        spin_up_atom = [expect(tensor(qeye(3), qeye(N), basis(2, 0) * basis(2, 0).dag()), final_state_atom)]

        atom_spin_down_list.append(spin_down_atom)
        atom_spin_up_list.append(spin_up_atom)
    

    return results, excitation, spin_down_list, spin_up_list, atom_spin_down_list, atom_spin_up_list



def odf_change_B(b_field_gauss, w_mols, heights, 
                 x_data, y_data,
                 motional_ground, times, times_atom, 
                 detunings, N, rabi_rate, rabi_rate_molecule, 
                 rabi_rate_atom, var_time, coherence, return_results):

    spin_up = basis(3, 0)  # Up state for spin
    spin_down = basis(3, 1)  # Down state
    auxiliary_state = basis(3, 2)

    spin_up_atom = basis(2,0)
    spin_down_atom = basis(2, 1)

    x = x_op_3mol_atom(N)
    p = p_op_3mol_atom(N)

    final_time = times[-1]

    for i in range(len(b_field_gauss)):
        B_field = b_field_gauss[i]
        w_mol = w_mols[i]
        coeff = heights[i]

        print(f"B field = {B_field} G, w_mol = {w_mol*1e6:.2f} Hz, population coefficient = {coeff:.4f}")

        molecule_state = np.sqrt(coeff)*spin_down + np.sqrt(1-coeff)*auxiliary_state

        # INITIAL STATE WAVEFUNCTION (tensor(spin_state, motion_state))
        psi0 = tensor(molecule_state, motional_ground, spin_down_atom) 
        
        results_odf, exc_mol_odf, spin_down_mol_odf, spin_up_mol_odf, spin_down_atom_odf, spin_up_atom_odf = molec3_atom_evolution(psi0, Ham="odf", times=times, times_atom=times_atom, det=detunings, x=x, p=p, N=N, rr = rabi_rate, rr_mol = None, w_mol = None, rr_atom = rabi_rate_atom, var_time=var_time, coherence=coherence, return_results=return_results)
        results_bsb, exc_mol_bsb, spin_down_mol_bsb, spin_up_mol_bsb, spin_down_atom_bsb, spin_up_atom_bsb = molec3_atom_evolution(psi0, Ham="bsb", times=times, times_atom=times_atom, det=detunings, x=x, p=p, N=N, rr = None, rr_mol = rabi_rate_molecule, w_mol = w_mol, rr_atom = rabi_rate_atom, var_time=var_time, coherence=coherence, return_results=return_results)
        results_tot, exc_mol_tot, spin_down_mol_tot, spin_up_mol_tot, spin_down_atom_tot, spin_up_atom_tot = molec3_atom_evolution(psi0, Ham="tot", times=times, times_atom=times_atom, det=detunings, x=x, p=p, N=N, rr = rabi_rate, rr_mol = rabi_rate_molecule, w_mol = w_mol, rr_atom = rabi_rate_atom, var_time=var_time, coherence=coherence, return_results=return_results)

        plot_atom_mol3_evolution(x_data, y_data, final_time, w_mol, rabi_rate_molecule, rabi_rate, detunings, exc_mol_odf, exc_mol_bsb, exc_mol_tot, spin_up_atom_odf, spin_up_atom_bsb, spin_up_atom_tot, B=B_field)




def odf_3mol_data(N, coeff, motional_ground, 
                  times, times_atom, detunings, 
                  rabi_rate, rabi_rate_atom, 
                  var_time, coherence, return_results, 
                  x_data, y_data):

    spin_up = basis(3, 0)  # Up state for spin
    spin_down = basis(3, 1)  # Down state
    auxiliary_state = basis(3, 2)

    spin_up_atom = basis(2,0)
    spin_down_atom = basis(2, 1)

    x = tensor(qeye(3), create(N) + destroy(N), qeye(2)) / 2
    p = tensor(qeye(3), 1j * (create(N) - destroy(N)), qeye(2)) / 2


    molecule_state = np.sqrt(coeff)*spin_down + np.sqrt(1-coeff)*auxiliary_state

    # INITIAL STATE WAVEFUNCTION (tensor(spin_state, motion_state))
    psi0 = tensor(molecule_state, motional_ground, spin_down_atom) 

    results_odf, exc_mol_odf, spin_down_mol_odf, spin_up_mol_odf, spin_down_atom_odf, spin_up_atom_odf = molec3_atom_evolution(psi0, Ham="odf", times=times, times_atom=times_atom, det=detunings, x=x, p=p, N=N, rr = rabi_rate, rr_mol = None, w_mol = None, rr_atom = rabi_rate_atom, var_time=var_time, coherence=coherence, return_results=return_results)

    plot_odf_3mol_data(detunings, spin_up_atom_odf, x_data, y_data)



###################################################################################
########################## MOLECULE + MOTION + ION ################################
###################################################################################


##### MOLECULE-ATOM EVOLUTIONS #####


def molec_atom_evolution(psi0, Ham, times, times_atom, det, x, p, N, rr = None, rr_mol = None, w_mol = None, rr_atom = None):
    x_vals = []
    p_vals = []
    prob = []
    excitation = []
    spin_down_list = []
    spin_up_list = []
    atom_spin_up_list = []
    atom_spin_down_list = []


    opts = Options(atol=1e-10, rtol=1e-8)

    results = []


    for delta in det:

        init_psi = psi0
        if Ham == "bsb":
            args = {'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2*np.pi * w_mol, 'N': N}
            result = mesolve(H_mol_atom_bsb, init_psi, times, [], [], args= args, options=opts)
        elif Ham == "tot":
            args = {'rabi_rate': rr, 'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2*np.pi * w_mol, 'N': N}
            result = mesolve(H_mol_atom_tot, init_psi, times, [], [], args= args, options=opts)
        elif Ham == "odf":
            args = {'rabi_rate': rr, 'd': delta, 'N': N}
            result = mesolve(H_mol_atom_odf, init_psi, times, [], [], args= args, options=opts)
        else:
            print("Invalid type")


        results.append(result)

        # I take the final state and its population in |0>
        final_state_mol = result.states[-1]

        

        final_state_mol_p0 = final_state_mol.ptrace(1)[0,0]
        norm = np.abs(final_state_mol_p0)**2
        prob.append(1 - norm)

        # I compute the expectation on the number operator for the motional subsystem. Then take the last step in the simulation
        excitation_prob = expect(tensor(qeye(2), num(N), qeye(2)), final_state_mol)
        excitation.append(excitation_prob)

        x_final = expect(x, final_state_mol)
        p_final = expect(p, final_state_mol)
        x_vals.append(x_final)
        p_vals.append(p_final)

        pop_spin_down = [expect(tensor(basis(2, 1) * basis(2, 1).dag(), qeye(N), qeye(2)), final_state_mol)]
        pop_spin_up = [expect(tensor(basis(2, 0) * basis(2, 0).dag(), qeye(N), qeye(2)), final_state_mol)]

        spin_down_list.append(pop_spin_down)
        spin_up_list.append(pop_spin_up)

        # RBR ATOM READOUT
        args_atom = {'rabi_rate_atom': rr_atom, 'N': N}
        result_atom = mesolve(H_mol_atom_rsb, final_state_mol, times_atom, [], [], args=args_atom)

        final_state_atom = result_atom.states[-1]

        spin_down_atom = [expect(tensor(qeye(2), qeye(N), basis(2, 1) * basis(2, 1).dag()), final_state_atom)]
        spin_up_atom = [expect(tensor(qeye(2), qeye(N), basis(2, 0) * basis(2, 0).dag()), final_state_atom)]

        atom_spin_down_list.append(spin_down_atom)
        atom_spin_up_list.append(spin_up_atom)

    

    return results, excitation, prob, x_vals, p_vals, spin_down_list, spin_up_list, atom_spin_down_list, atom_spin_up_list





###################################################################################
########################## MOLECULE + MOTION ######################################
###################################################################################




##### MOLECULE EVOLUTIONS #####

def molec_evolution(psi0, Ham, times, det, x, p, N, rr = None, rr_mol = None, w_mol = None):
    x_vals = []
    p_vals = []
    prob = []
    excitation = []
    spin_down_list = []
    spin_up_list = []


    for delta in det:

        init_psi = psi0
        if Ham == "bsb":
            args = {'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2*np.pi * w_mol, 'N': N}
            result = mesolve(H_mol_bsb, init_psi, times, [], [], args= args)
        elif Ham == "tot":
            args = {'rabi_rate': rr, 'rabi_rate_molecule': rr_mol, 'd': delta, 'w_mol': 2*np.pi * w_mol, 'N': N}
            result = mesolve(H_mol_tot, init_psi, times, [], [], args= args)
        elif Ham == "odf":
            args = {'rabi_rate': rr, 'd': delta, 'N': N}
            result = mesolve(H_mol_odf, init_psi, times, [], [], args= args)
        else:
            print("Invalid type")


        # I take the final state and its population in |0>
        final_state_mol = result.states[-1]

        final_state_mol_p0 = final_state_mol.ptrace(1)[0,0]
        norm = np.abs(final_state_mol_p0)**2
        prob.append(1 - norm)

        # I compute the expectation on the number operator for the motional subsystem. Then take the last step in the simulation
        excitation_prob = expect(tensor(qeye(2), num(N)), final_state_mol)
        excitation.append(excitation_prob)

        x_final = expect(x, final_state_mol)
        p_final = expect(p, final_state_mol)
        x_vals.append(x_final)
        p_vals.append(p_final)

        pop_spin_down = [expect(tensor(basis(2, 1) * basis(2, 1).dag(), qeye(N)), final_state_mol)]
        pop_spin_up = [expect(tensor(basis(2, 0) * basis(2, 0).dag(), qeye(N)), final_state_mol)]

        spin_down_list.append(pop_spin_down)
        spin_up_list.append(pop_spin_up)
    

    return excitation, prob, x_vals, p_vals, spin_down_list, spin_up_list


