import numpy as np
from qutip import *
from RAP.rap_operators import RAP_sigmap_2




def RAP_bsb_H_2levels(t, args) :
    """
    > Time-dependent Hamiltonian for a 2-level molecule interacting with a BSB (blue sideband) laser.

    Includes both resonant and optionally off-resonant carrier contributions. Time dependence comes from:
    - Gaussian envelope in the Rabi rate,
    - Time-dependent detuning.

    Parameters:
    -----------
    t : float
        Time during the simulation.
    args : dict
        Dictionary of parameters including trap frequency, Lamb-Dicke factor, detunings, etc.

    Returns:
    --------
    Qobj
        Hamiltonian operator at time `t`.
    """


    n_motional = args['n_motional']     # number of motional states
    j = args['j']                     # total angular momentum of the molecule
    coupling = args['coupling']     # coupling constant of the transition
    w_mol = args['w_mol']           # molecular transition frequency
    final_time = args['final_time']
    D = args['D']           
    sigma = args['sigma']
    T = args['T']     
    lamb_dicke = args['lamb_dicke']    
    off_resonant = args['off_resonant'] 
    w_trap = args['trap_freq']     

    sigma_plus_mol = tensor(RAP_sigmap_2(), qeye(n_motional))
    a = tensor(qeye(2), destroy(n_motional))

    # Time dependent RAP variables
    rr = args['rabi_rate']   # Rabi rate of the transition
    rr = rr * np.exp(-(t - T/2)**2/(2*sigma**2))  
    laser_detuning = args['laser_detuning']     # laser detuning   

    det_bsb = laser_detuning - w_mol      # detuning of the molecular transition

    delta_t_bsb = D/T * (t - T/2) + det_bsb

    H_term_bsb = sigma_plus_mol * a.dag() * np.exp(-1j*delta_t_bsb*t)

    H_bsb = rr * np.abs(coupling) /2 * (H_term_bsb + H_term_bsb.dag())

    if off_resonant:
        if lamb_dicke is None:
            raise ValueError("lamb_dicke must be provided when off_resonant is True")
        # Carrier off-resonant
        det_carrier = laser_detuning - (- w_trap + w_mol)  

        delta_t_carrier = D/T * (t - T/2) + det_carrier

        # CHECK THE SIGNS
        H_term_carrier = sigma_plus_mol * np.exp(-1j*delta_t_carrier*t)
        H_carrier = rr * np.abs(coupling) /2 / lamb_dicke * (H_term_carrier + H_term_carrier.dag())
    
        # return H_bsb + H_carrier
        return H_bsb + H_carrier
    else:
        return H_bsb 




def RAP_carrier_H_2levels(t, args):
    """
    > Time-dependent Hamiltonian for a 2-level molecule interacting with a carrier transition.

    Like `RAP_bsb_H_2levels`, but without coupling to motion (i.e., no ladder operators for motional states).

    Parameters:
    -----------
    t : float
        Time during the simulation.
    args : dict
        Dictionary of parameters for the RAP pulse and molecular level structure.

    Returns:
    --------
    Qobj
        Hamiltonian operator at time `t`.
    """


    n_motional = args['n_motional']     # number of motional states
    j = args['j']                     # total angular momentum of the molecule
    coupling = args['coupling']     # coupling constant of the transition
    w_mol = args['w_mol']           # molecular transition frequency
    final_time = args['final_time']
    D = args['D']           
    sigma = args['sigma']
    T = args['T']           

    sigma_plus_mol = tensor(RAP_sigmap_2(), qeye(n_motional))
    a = tensor(qeye(2), destroy(n_motional))

    # Time dependent RAP variables
    rr = args['rabi_rate']   # Rabi rate of the transition
    rr = rr * np.exp(-(t - T/2)**2/(2*sigma**2))  
    laser_detuning = args['laser_detuning']     # laser detuning   

    det_carrier = laser_detuning - w_mol

    delta_t_carrier = D/T * (t - T/2) + det_carrier

    # CHECK THE SIGNS
    H_term_carrier = sigma_plus_mol * np.exp(-1j*delta_t_carrier*t)
    H_carrier = rr * np.abs(coupling) /2 * (H_term_carrier + H_term_carrier.dag())
    
    return H_carrier



def RAP_rabiflop_H_2levels(t, args):
    """
    > Hamiltonian for Rabi flop (time-independent frequency, no chirp).

    Similar to `RAP_bsb_H_2levels`, but assumes constant detuning and no Gaussian envelope. Can include off-resonant carrier terms if specified.

    Parameters:
    -----------
    t : float
        Time during the simulation.
    args : dict
        Dictionary of parameters, including detuning and coupling strength.

    Returns:
    --------
    Qobj
        Hamiltonian operator at time `t`.
    """


    n_motional = args['n_motional']     # number of motional states
    j = args['j']                     # total angular momentum of the molecule
    coupling = args['coupling']     # coupling constant of the transition
    w_mol = args['w_mol']           # molecular transition frequency
    final_time = args['final_time']
    D = args['D']           
    sigma = args['sigma']
    T = args['T']        
    lamb_dicke = args['lamb_dicke']
    off_resonant = args['off_resonant']
    w_trap = args['trap_freq'] 

    sigma_plus_mol = tensor(RAP_sigmap_2(), qeye(n_motional))
    a = tensor(qeye(2), destroy(n_motional))

    # Time dependent RAP variables
    rr = args['rabi_rate']   # Rabi rate of the transition
    laser_detuning = args['laser_detuning']     # laser detuning   

    det_bsb = laser_detuning - w_mol      # detuning of the molecular transition

    H_term_bsb = sigma_plus_mol * a.dag() * np.exp(-1j*det_bsb*t)

    H_bsb = rr * np.abs(coupling) /2 * (H_term_bsb + H_term_bsb.dag())

    if off_resonant:
        if lamb_dicke is None:
            raise ValueError("lamb_dicke must be provided when off_resonant is True")
        # # Carrier off-resonant
        det_carrier = laser_detuning - (- w_trap + w_mol)  

        delta_t_carrier = D/T * (t - T/2) + det_carrier

        # CHECK THE SIGNS
        H_term_carrier = sigma_plus_mol * np.exp(-1j*delta_t_carrier*t)
        H_carrier = rr * np.abs(coupling) /2 / lamb_dicke * (H_term_carrier + H_term_carrier.dag())
        
        return H_bsb + H_carrier
    else: 
        return H_bsb 
    


