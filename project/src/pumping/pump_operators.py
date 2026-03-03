import numpy as np
from qutip import *


def sigmap_n(init_index, final_index, n_internal):
    """
    Function to create the sigmap operator in the internal space.
    The size of the matrix is the number of internal states.
    init_index: index of the initial state
    final_index: index of the final state 
    """

    sigmap_n = np.zeros((n_internal, n_internal))
    sigmap_n[final_index, init_index] = 1.0

    op_mol = Qobj(sigmap_n)
    return op_mol



def collapse_cooling_op(decay_rate, n_internal, n_motional):
    """
    collapse operator for the cooling process.
    The operator is a tensor product of the identity operator in the internal space and the lowering operator in the motional space times the square root of the decay rate.

    """

    op = tensor(qeye(n_internal), np.sqrt(decay_rate)*destroy(n_motional))  # decay operator in the motional space

    return [op]





