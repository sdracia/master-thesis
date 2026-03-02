import numpy as np
from qutip import *




def RAP_collapse_cooling_op(decay_rate, n_motional):
    """
    > Collapse operator for the cooling process in the RAP simulation.

    This operator acts on the motional degree of freedom, representing spontaneous emission or phonon loss.
    It is defined as the tensor product of:
    - The identity operator in the 2-level internal state space.
    - The lowering operator `a` in the motional space scaled by √(decay_rate).

    Parameters:
    -----------
    decay_rate : float
        The decay rate of the cooling process.
    n_motional : int
        Dimension of the motional Hilbert space (number of Fock states).

    Returns:
    --------
    list of Qobj
        A list containing the single collapse operator.
    """

    op = tensor(qeye(2), np.sqrt(decay_rate)*destroy(n_motional))  # decay operator in the motional space

    return [op]




def RAP_sigmap_2():
    """
    > Returns the raising operator σ⁺ for a 2-level system.

    This operator acts only on the internal molecular degree of freedom.

    Returns:
    --------
    Qobj
        The 2x2 raising operator.
    """

    sigmap_n = np.zeros((2, 2))
    sigmap_n[1, 0] = 1.0

    op_mol = Qobj(sigmap_n)
    return op_mol


