import numpy as np

from molecules.molecule import Molecule




def j_distribution(self, molecule: Molecule , j_max: int) -> np.ndarray:

    list_pop = []

    for j_val in range(0, j_max + 1):
        indices_in_j = molecule.state_df.index[molecule.state_df["j"] == j_val]
        total_in_j = molecule.state_df.loc[indices_in_j, "state_dist"].sum()

        list_pop.append(total_in_j)

    return np.array(list_pop)
