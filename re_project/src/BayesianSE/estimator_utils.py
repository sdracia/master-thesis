import numpy as np




# prior is the column in the dataframe in the molecule
def init_prior(self):
    self.prior = self.model.state_df["state_dist"]



def stop_condition(self, posterior):
    return any(p > 0.9 for p in posterior)



def cross_entropy(self, simulator):

    posteriors = [np.array(entry["posterior"]) for entry in self.history_list]

    cross_entropies = []
    for i in range(len(simulator.history_list[1:])):
        index = simulator.history_list[i+1][0]
        posterior = posteriors[i]
        cross_entropy = -np.log(posterior[index] + 1e-11)  

        
        if cross_entropy < 0 and np.isclose(cross_entropy, 0, atol=1e-10):
            cross_entropy = 1e-11

        cross_entropies.append(cross_entropy)

    for i, ce in enumerate(cross_entropies):
        if ce < 0:
            raise ValueError(f"Negative Cross Entropy found at index {i}: {ce}")


    return cross_entropies