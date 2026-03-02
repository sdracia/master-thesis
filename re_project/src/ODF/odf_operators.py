from qutip import *
import numpy as np


###################################################################################
########################## MOLECULE + MOTION + ION ################################
########################## MOLECULE WITH 3 LEVELS  ################################
###################################################################################

# sigma+ operator which goes from |1> → |0>
sigmap_3 = Qobj(np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0]
]))

# sigma- operator which goes from |0> → |1>
sigmam_3 = Qobj(np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0]
]))

sigmaz_3 = Qobj(np.array([
    [1, 0, 0],
    [0,-1, 0],
    [0, 0, 0]
]))


def decay(coherence_time):
    "decay operator, with coh_time being the spin_coherence time in us"
    return np.sqrt(1/(coherence_time*2)) * sigmaz_3


def x_op_3mol_atom(N):
    "x operator for the motion"
    return tensor(qeye(3), create(N) + destroy(N), qeye(2)) / 2

def p_op_3mol_atom(N):
    "p operator for the motion"
    return tensor(qeye(3), 1j * (create(N) - destroy(N)), qeye(2)) / 2




###################################################################################
########################## MOLECULE + MOTION + ION ################################
###################################################################################


def x_op_mol_atom(N):
    "x operator for the motion"
    return tensor(qeye(2), create(N) + destroy(N), qeye(2)) / 2


def p_op_mol_atom(N):
    "p operator for the motion"
    return tensor(qeye(2), 1j * (create(N) - destroy(N)), qeye(2)) / 2



###################################################################################
########################## MOLECULE + MOTION ######################################
###################################################################################


def x_op_molecule(N):
    "x operator for the motion"
    return tensor(qeye(2), create(N) + destroy(N)) / 2

def p_op_molecule(N):
    "p operator for the motion"
    return tensor(qeye(2), 1j * (create(N) - destroy(N))) / 2