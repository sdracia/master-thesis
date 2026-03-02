import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from qutip import *


from saving import save_figure_in_images




def compute_transitions(molecule, j_max):
    """
    Description
    -----------
    Compute the following transitions dataframe:
    - left_states_transitions: the transitions between the left states of the lower manifolds and the state in the upper ones, with dm = -1, for all the manifolds in the molecule.
    - left_to_right_transitions: the transitions between the left states of the lower manifolds and the state in the upper ones, with dm = +1, for all the manifolds in the molecule.
    - signature_transitions_left: the transitions between the penultimate left states of the upper manifolds and the state in the same upper ones, with dm = -1, for all the manifolds in the molecule.
    - signature_transitions_right: the transitions between the penultimate left states of the upper manifolds and the state in the lower ones, with dm = +1, for all the manifolds in the molecule.

    Parameters
    ----------
    molecule : Molecule
        The molecule object containing the transition data.
    j_max : int
        The maximum value of j to consider for transitions.
    
    Returns
    -------
    left_states_transitions : pd.DataFrame
    left_to_right_transitions : pd.DataFrame
    signature_transitions_left : pd.DataFrame
    signature_transitions_right : pd.DataFrame

    """
    left_states_transitions = []
    left_to_right_transitions = []
    signature_transitions_left = []
    signature_transitions_right = []

    for j_val in range(0, j_max + 1):

        transitions_in_j = molecule.transition_df[molecule.transition_df["j"] == j_val]

        filtered_transitions = transitions_in_j[(transitions_in_j["xi1"] == True) & (transitions_in_j["xi2"] == False)]

        signature_transition_left = transitions_in_j.iloc[0]
        signature_transitions_left.append(signature_transition_left)
        
        if not filtered_transitions.empty:
            left_state_trans = filtered_transitions.iloc[0]
            left_states_transitions.append(left_state_trans)
            
        
        if len(transitions_in_j) > 2:
            left_to_right_trans = transitions_in_j.iloc[2]
            left_to_right_transitions.append(left_to_right_trans)

            signature_transition_right = filtered_transitions.iloc[1]
            signature_transitions_right.append(signature_transition_right)

    left_states_transitions = pd.DataFrame(left_states_transitions)
    left_to_right_transitions = pd.DataFrame(left_to_right_transitions)
    signature_transitions_left = pd.DataFrame(signature_transitions_left)
    signature_transitions_right = pd.DataFrame(signature_transitions_right)

    # Reversing the order of the DataFrames
    # left_states_transitions = pd.DataFrame(left_states_transitions).iloc[::-1].reset_index(drop=True)
    # left_to_right_transitions = pd.DataFrame(left_to_right_transitions).iloc[::-1].reset_index(drop=True)
    # signature_transitions_left = pd.DataFrame(signature_transitions_left).iloc[::-1].reset_index(drop=True)
    # signature_transitions_right = pd.DataFrame(signature_transitions_right).iloc[::-1].reset_index(drop=True)


    return left_states_transitions, left_to_right_transitions, signature_transitions_left, signature_transitions_right





def RAP_args(dataframe, is_minus, n_motional, rabi_rate, laser_detuning, times, T, D, sigma, trap_freq = None, lamb_dicke = None, off_resonant = False):
    """
    > Prepares the argument dictionary for RAP simulations.

    Builds a list of dictionaries, each corresponding to one transition manifold (`j`), containing the relevant
    parameters for the simulation: detuning, coupling, Rabi rate, etc.

    Additionally, augments the input DataFrame with state labeling used to track population transfers.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        Contains info on energy levels and coupling strengths for different `j`.
    is_minus : bool
        Determines the directionality of the pulse (based on manifold).
    n_motional : int
        Number of motional states.
    rabi_rate : float
        Maximum Rabi frequency of the laser field.
    laser_detuning : float
        Detuning of the laser in angular frequency units.
    times : array
        Time array for the simulation.
    T : float
        Total duration of the RAP pulse.
    D : float
        Frequency sweep parameter (slope).
    sigma : float
        Width of the Gaussian amplitude envelope.
    trap_freq : float, optional
        Motional trap frequency (for off-resonant terms).
    lamb_dicke : float, optional
        Lamb-Dicke parameter.
    off_resonant : bool, default=False
        Whether to include off-resonant carrier terms.

    Returns:
    --------
    args : dict
        Dictionary with key `'terms'` holding simulation parameters for each manifold.
    rescaled_df : pd.DataFrame
        Input dataframe with added labels for internal state indexing.
    """

    H_terms_args = []

    rescaled_df = dataframe.reset_index(drop=True).copy()

    # For each manifold, label_1 is the state in the lower manifold. I swap the population from here
    rescaled_df['label_1'] = rescaled_df.index * 2        
    rescaled_df['label_2'] = rescaled_df.index * 2 + 1

    for _, row in rescaled_df.iterrows():

        j = int(row["j"])
        coupling = row["coupling"]

        # since the detuning are computed wrt is_minus = True, then if is_minus = False, the values of the detunings are inverted
        if is_minus:
            w_mol = row["energy_diff"] * 1e-3
        else:
            w_mol = -row["energy_diff"] * 1e-3

        term_args = {'n_motional': int(n_motional), 
                'j': int(j),
                'rabi_rate': rabi_rate,
                'coupling': np.abs(coupling), 
                'w_mol': 2* np.pi * w_mol, 
                'laser_detuning': laser_detuning,
                'trap_freq': trap_freq,
                'lamb_dicke': lamb_dicke,
                'off_resonant': off_resonant,
                'final_time': times[-1],
                'T': T,
                'D': D,
                'sigma': sigma}

        H_terms_args.append(term_args)
    
    args = {'terms': H_terms_args}

    return args, rescaled_df


def RAP_dm(term_args, n_motional, from_simulation = False, sideband = True, init_pop_list = [0.5, 0.5]):
    """
    > Builds the initial density matrix for a given RAP transition.

    If `from_simulation` is True, logic should be added to extract a reduced density matrix from a prior result.
    Otherwise, builds a completely mixed state as tensor product of:
    - The internal 2-level state (ground/excited or average),
    - The motional vacuum state.

    Parameters:
    -----------
    term_args : dict
        Simulation parameters for the specific transition.
    n_motional : int
        Number of motional states.
    from_simulation : bool, default=False
        If True, loads initial state from previous simulation (not implemented).
    sideband : bool, default=True
        Whether this is a sideband transition (else carrier transition).
    init_pop_list : list of float, default=[0.5, 0.5]
        List specifying the population of each internal state (must sum to 1).
        For a qubit: [p0, p1] where p0 is population of |0⟩, p1 of |1⟩.

    Returns:
    --------
    rho : Qobj
        Initial density matrix.
    """

    # basis(2,0) is the ground state, basis(2,1) is the excited state

    if from_simulation:
        j = term_args['j']
        population_init = term_args['population_init']
        population_fin = term_args['population_fin']

        ## STILL TO FIGURE OUT HOW TO DO.
        # 1. i take the rho and then consider the reduced density matrix
        # 2. I consider only the 2 population
        #
        # TO THINK. with the molecule I have the motional state that is shared between all the manifolds. 
        # Simulating the BSB and then cooling i dont know if it is correct.

    else:
        # basis(2,0) is the ground state, basis(2,1) is the excited state
        if sideband:

            assert np.isclose(sum(init_pop_list), 1.0), "Populations must sum to 1"
            assert len(init_pop_list) == 2, "Only two-level systems are supported"

            states = [basis(2, i) for i in range(2)]
            rho_internal = sum([p * ket2dm(state) for p, state in zip(init_pop_list, states)])


            index_motional = 0
            psi_motional = basis(n_motional, index_motional)
            rho_motional = ket2dm(psi_motional)

        else:
            state = basis(2, 0)
            rho_internal = ket2dm(state)

            index_motional = 0
            psi_motional = basis(n_motional, index_motional)
            rho_motional = ket2dm(psi_motional)
            
        rho = tensor(rho_internal, rho_motional)

    print("Density Matrix created")
    return rho



def chirp_envelope(args, times, T, D, sigma, rabi_rate, laser_detuning, 
                   j_plot = None, savetext = "rap_pulse"):

    """
    -------------------------------------------------------------------------------
     Plot the time-dependent Rabi rate and frequency detuning (chirp envelope)
     as well as the adiabaticity condition for a RAP pulse.

     Parameters
     ----------
     args : dict
         Dictionary containing the system configuration, including 'terms' with:
         - j : float, angular momentum quantum number
         - w_mol : float, molecular resonance frequency (in Hz)
     times : ndarray
         Array of time points at which the envelope is evaluated.
     T : float
         Total pulse duration in seconds.
     D : float
         Frequency sweep range (Hz) for the linear chirp.
     sigma : float
         Width of the Gaussian pulse envelope (seconds).
     rabi_rate : float
         Peak Rabi frequency (Hz).
     laser_detuning : float
         Initial detuning of the laser with respect to resonance (Hz).

     Returns
     -------
     None
         Displays two plots:
         - Left: time-dependent Rabi rate and frequency detuning
         - Right: adiabaticity condition vs time
    -------------------------------------------------------------------------------
    """

    available_j = [term["j"] for term in args["terms"]]
    J_max = max(available_j)

    if j_plot is None:
        j_to_plot = available_j

    elif isinstance(j_plot, int):
        j_to_plot = [j_plot]

    else:
        j_to_plot = list(j_plot)

    for j in j_to_plot:
        if j not in available_j:
            raise ValueError(
                f"Requested J={j} not available. "
                f"Available J range: {min(available_j)} to {J_max}"
            )

    for arg in args['terms']:
        j = arg['j']

        if j not in j_to_plot:
            continue

        w_mol = arg['w_mol']


        omega_t = rabi_rate * np.exp(-(times - T/2)**2/(2*sigma**2))

        delta_t = D/T * (times - T/2) + laser_detuning - w_mol

        omega_eff = np.sqrt(omega_t**2 + delta_t**2)
        theta = np.abs(np.arctan2(omega_t, delta_t) / 2)

        paper_adiabaticity = np.sqrt(omega_t**2 * (times - T/2)**2 / sigma**4 + D**2/T**2) / (omega_t**2 + delta_t**2)

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 4))  # 2 subplots side-by-side

        # === Left subplot: Rabi Envelope and Chirp Frequency (unchanged) ===
        color1 = 'tab:blue'
        ax1.set_xlabel('Time (ms)', fontsize=20)
        ax1.set_ylabel('Rabi rate (kHz)', color=color1, fontsize=20)
        ax1.plot(times*1e-3, omega_t*1e3/(2*np.pi), color=color1, label='Rabi Rate ')
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=20)
        ax1.tick_params(axis='x', labelcolor="black", labelsize=20)

        # ax1.axvline(x=final_time/2, color='gray', linestyle='--', linewidth=1)
        ax1.grid()

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Detuning (kHz)', color=color2, fontsize=20)
        ax2.plot(times*1e-3, delta_t*1e3/(2*np.pi), color=color2, label='Detuning')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=20)
        ax2.tick_params(axis='x', labelcolor="black", labelsize=20)


        # === Right subplot: Adiabaticity Conditions ===
        ax3.set_title("Adiabaticity Conditions")
        # ax3.plot(times, system_adiabaticity, label="System Adiabaticity", color="green")
        # ax3.plot(times, local_adiabaticity, label="Local Adiabaticity", color="purple", linestyle="--")
        ax3.plot(times, paper_adiabaticity, label="Local Adiabaticity", color="blue", linestyle="-")
        ax3.set_xlabel("Time (ms)", fontsize=20)
        ax3.set_ylabel("Adiabaticity Value", fontsize=20)
        ax3.grid()
        ax3.legend()


        # Title and layout
        plt.title(fr'Time dependence of $\Omega (t)$ and $\Delta (t)$, J = {j}', fontsize=25)
        # plt.tick_params(axis='both', which='major', labelsize=25)
        plt.tick_params(axis='both', labelsize=20)


        fig.tight_layout()

        filename = f"{savetext}_j{j}.svg"
        save_figure_in_images(fig, filename)

        plt.show()




def animate_bloch(result, times, duration=0.1, color='r', animfilename='anim_gif'):
    '''Function for analyzing evolution of a single qubit on the Bloch sphere.'''

    sigmax_op = - sigmax()
    sigmay_op = - sigmay()
    sigmaz_op = - sigmaz()

    vectors = []
    pxs, pys, pzs = [], [], []

    for state in result.states:
        rho = state.ptrace(0)
        
        print(rho)
        purity = (rho*rho).tr()
        # print(purity)

        vector = [expect(sigmax_op, rho), expect(sigmay_op, rho), expect(sigmaz_op, rho)]
        print(vector)
        vectors.append(vector)
        pxs.append(vector[0])
        pys.append(vector[1])
        pzs.append(vector[2])

    # Determine how many frames to generate

    length = len(times)

    # Create Bloch sphere animation
    b = Bloch()
    b.view = [-40, 30]
    b.vector_color = [color]
    b.point_color = [color]
    b.point_marker = ['o']
    b.point_size = [30]

    images = []
    for i in range(length):
        b.clear()
        b.add_vectors(vectors[i])
        b.add_points([pxs[:i+1], pys[:i+1], pzs[:i+1]], meth='l')
        filename = 'temp_file.png'
        b.save(filename)
        images.append(imageio.imread(filename))

    imageio.mimsave('images/' + animfilename + '.gif', images, duration=duration)

