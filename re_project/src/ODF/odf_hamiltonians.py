from qutip import *
import numpy as np


from ODF.odf_operators import sigmap_3, sigmam_3


###################################################################################
########################## MOLECULE + MOTION + ION ################################
########################## MOLECULE WITH 3 LEVELS  ################################
###################################################################################


# ODF
def H_mol3_atom_odf(t, args) :
    N = args['N']
    x = tensor(qeye(3), create(N) + destroy(N), qeye(2)) / 2
    p = tensor(qeye(3), 1j * (create(N) - destroy(N)), qeye(2)) / 2


    rabi_rate = args['rabi_rate']
    d = args['d']

    return x * rabi_rate* np.sin(d*t) + p * rabi_rate * np.cos(d*t)



# BSB
def H_mol3_atom_bsb(t, args) :
    N = args['N']
    sigma_plus_mol = tensor(sigmap_3, qeye(N), qeye(2))
    sigma_minus_mol = tensor(sigmam_3, qeye(N), qeye(2))
    a = tensor(qeye(3), destroy(N), qeye(2))

    rabi_rate_molecule = args['rabi_rate_molecule'] 
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol 

    
    return rabi_rate_molecule /2 * (sigma_plus_mol * a.dag() * np.exp(-1j*d_mol*t) + sigma_minus_mol * a * np.exp(1j*d_mol*t))


# BSB + ODF
def H_mol3_atom_tot(t, args) :
    N = args['N']
    sigma_plus_mol = tensor(sigmap_3, qeye(N), qeye(2))
    sigma_minus_mol = tensor(sigmam_3, qeye(N), qeye(2))
    a = tensor(qeye(3), destroy(N), qeye(2))
    x = tensor(qeye(3), create(N) + destroy(N), qeye(2)) / 2
    p = tensor(qeye(3), 1j * (create(N) - destroy(N)), qeye(2)) / 2

    rabi_rate = args['rabi_rate']
    rabi_rate_molecule = args['rabi_rate_molecule'] 
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol 
    # ODF hamiltonian + BSB hamiltoninan
    return (x * rabi_rate* np.sin(d*t) + p * rabi_rate * np.cos(d*t)) + (rabi_rate_molecule / 2 * (sigma_plus_mol * a.dag() * np.exp(-1j*d_mol*t) + sigma_minus_mol * a * np.exp(1j*d_mol*t))) 


# RSB
def H_mol3_atom_rsb(t, args):
    N = args['N']
    sigma_plus_atom = tensor(qeye(3), qeye(N), sigmap())
    sigma_minus_atom = tensor(qeye(3), qeye(N), sigmam())
    a = tensor(qeye(3), destroy(N), qeye(2))

    rabi_rate_atom = args['rabi_rate_atom']
    return rabi_rate_atom / 2 * (sigma_minus_atom * a.dag() + sigma_plus_atom * a)




###################################################################################
########################## MOLECULE + MOTION + ION ################################
###################################################################################

##### HAMILTONIAN DEFINITIONS #####


# ODF
def H_mol_atom_odf(t, args) :
    N = args['N']
    x = tensor(qeye(2), create(N) + destroy(N), qeye(2)) / 2
    p = tensor(qeye(2), 1j * (create(N) - destroy(N)), qeye(2)) / 2


    rabi_rate = args['rabi_rate']
    d = args['d']

    return x * rabi_rate* np.sin(d*t) + p * rabi_rate * np.cos(d*t)


# BSB
def H_mol_atom_bsb(t, args) :
    N = args['N']
    sigma_plus_mol = tensor(sigmap(), qeye(N), qeye(2))
    sigma_minus_mol = tensor(sigmam(), qeye(N), qeye(2))
    a = tensor(qeye(2), destroy(N), qeye(2))

    rabi_rate_molecule = args['rabi_rate_molecule'] 
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol 
    
    return rabi_rate_molecule /2 * (sigma_plus_mol * a.dag() * np.exp(-1j*d_mol*t) + sigma_minus_mol * a * np.exp(1j*d_mol*t))


# BSB + ODF
def H_mol_atom_tot(t, args) :
    N = args['N']
    sigma_plus_mol = tensor(sigmap(), qeye(N), qeye(2))
    sigma_minus_mol = tensor(sigmam(), qeye(N), qeye(2))
    a = tensor(qeye(2), destroy(N), qeye(2))
    x = tensor(qeye(2), create(N) + destroy(N), qeye(2)) / 2
    p = tensor(qeye(2), 1j * (create(N) - destroy(N)), qeye(2)) / 2

    rabi_rate = args['rabi_rate']
    rabi_rate_molecule = args['rabi_rate_molecule'] 
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol 
    # ODF hamiltonian + BSB hamiltoninan
    return (x * rabi_rate* np.sin(d*t) + p * rabi_rate * np.cos(d*t)) + (rabi_rate_molecule / 2 * (sigma_plus_mol * a.dag() * np.exp(-1j*d_mol*t) + sigma_minus_mol * a * np.exp(1j*d_mol*t))) 


# RSB
def H_mol_atom_rsb(t, args):
    N = args['N']
    sigma_plus_atom = tensor(qeye(2), qeye(N), sigmap())
    sigma_minus_atom = tensor(qeye(2), qeye(N), sigmam())
    a = tensor(qeye(2), destroy(N), qeye(2))

    rabi_rate_atom = args['rabi_rate_atom']
    return rabi_rate_atom / 2 * (sigma_minus_atom * a.dag() + sigma_plus_atom * a)






###################################################################################
########################## MOLECULE + MOTION ######################################
###################################################################################

##### HAMILTONIAN DEFINITIONS #####

# ODF 
def H_mol_odf(t, args) :
    N = args['N']
    x = tensor(qeye(2), create(N) + destroy(N)) / 2
    p = tensor(qeye(2), 1j * (create(N) - destroy(N))) / 2
    rabi_rate = args['rabi_rate']
    d = args['d']

    return x * rabi_rate* np.sin(d*t) + p * rabi_rate * np.cos(d*t)


# ONLY BSB
def H_mol_bsb(t, args) :
    N = args['N']
    sigma_plus = tensor(sigmap(), qeye(N))
    sigma_minus = tensor(sigmam(), qeye(N))
    a = tensor(qeye(2), destroy(N))

    rabi_rate_molecule = args['rabi_rate_molecule'] 
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol 
    
    return rabi_rate_molecule / 2 * (sigma_plus * a.dag() * np.exp(-1j*d_mol*t) + sigma_minus * a * np.exp(1j*d_mol*t))


# BSB + ODF
def H_mol_tot(t, args) :
    N = args['N']
    x = tensor(qeye(2), create(N) + destroy(N)) / 2
    p = tensor(qeye(2), 1j * (create(N) - destroy(N))) / 2
    sigma_plus = tensor(sigmap(), qeye(N))
    sigma_minus = tensor(sigmam(), qeye(N))
    a = tensor(qeye(2), destroy(N))

    rabi_rate = args['rabi_rate']
    rabi_rate_molecule = args['rabi_rate_molecule'] 
    d = args['d']
    w_mol = args['w_mol']
    d_mol = d - w_mol 
    # ODF hamiltonian + BSB hamiltoninan
    return x * rabi_rate* np.sin(d*t) + p * rabi_rate * np.cos(d*t) + rabi_rate_molecule / 2 * (sigma_plus * a.dag() * np.exp(-1j*d_mol*t) + sigma_minus * a * np.exp(1j*d_mol*t))
