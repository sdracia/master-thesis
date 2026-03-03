import numpy as np




def state_initialization(self, j_max):
    j_values = np.arange(j_max + 1)
    j_probs = self.j_distribution(self.model, j_max)

    assert np.isclose(np.sum(j_probs), 1.0), "Distribution is not normalized."

    # Random sampling of J value according to the probability distribution
    j_sample = np.random.choice(j_values, p=j_probs)


    states_in_j = self.model.state_df.loc[self.model.state_df["j"] == j_sample].copy()

    # Indices in the DataFrame 
    indices = states_in_j.index.values

    # Extraction of the probability distribution, normalized with respect to the total population of that manifold 
    # (sum is not 1, sum is the thermal distribution of that manifold)
    probabilities = states_in_j["state_dist"].values / j_probs[j_sample]

    total = probabilities.sum()
    assert np.isclose(total, 1.0), "Distribution inside manifold is not normalized."

    # Random sampling of the index that labels the state according to the probability distribution of the states inside the manifold
    random_index = np.random.choice(indices, p=probabilities)

    selected_row = states_in_j.loc[random_index]
    j_val = selected_row["j"]
    m_val = selected_row["m"]
    xi_val = selected_row["xi"]

    print(f"Extracted level is idx={random_index}: J={j_val}, m={m_val}, xi={xi_val}", '\n')

    return random_index, j_val, m_val, xi_val



def new_state_index(self, col_sparse, rng=np.random.default_rng()):
    """
    Samples a row index from the sparse column `col_sparse`, treating it as a probability distribution.

    Args:
        col_sparse: a sparse column vector (CSC or CSR format, shape (N, 1))
        rng: random number generator (default: np.random.default_rng())

    Returns:
        A row index sampled according to the probability distribution defined by the column values.
    """
    # Extract non-zero row indices and their corresponding values
    row_indices = col_sparse.nonzero()[0]
    values = col_sparse.data

    # print(f"Row indices: {row_indices}. Values: {values}")

    # Normalize the values to form a probability distribution
    prob_dist = values / values.sum()

    # Sample a row index using the probability distribution
    chosen_idx = rng.choice(row_indices, p=prob_dist)
    
    return chosen_idx
