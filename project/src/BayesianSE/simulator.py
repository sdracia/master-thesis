from molecules.molecule import CaH, CaOH
from QLS.state_dist import States



from .simulator_utils import j_distribution
from .simulator_likelihood import likelihoods_simulator
from .simulator_dynamics import outcome_simulator
from .simulator_state import state_initialization, new_state_index






class BayesianSimulator:

    def __init__(self, 
                 model=None,
                 temperature = 300, 
                 b_field_gauss = 3.27, 
                 j_max = 50, 
                 false_positive_rate = 0.00,
                 false_negative_rate = 0.00):

        # Step 0. The molecule that is passed to the BayesianSimulator class has been already pumped and the state distribution
        #         reflect the one after the pumping + RAP (from the QuTip simulation)

        self.b_field_gauss = b_field_gauss
        self.j_max = j_max

        if model is None:
            model = CaOH.create_molecule_data(b_field_gauss=self.b_field_gauss, j_max=self.j_max)
        
        self.model = model

        if "state_dist" not in self.model.state_df.columns:
            states1 = States(molecule = self.model, temperature = temperature)

        self.temperature = temperature
        self.fpr = false_positive_rate
        self.fnr = false_negative_rate

        # Step 1. Initialization of the molecule in one manifold
        state_index, j_val, m_val, xi_val = self.state_initialization(j_max=self.j_max)

        # List to keep track of the molecule state after each pulse
        self.history_list = [[state_index, j_val, m_val, xi_val]]

        self.misfrequency = 0.0    

    j_distribution = j_distribution
    state_initialization = state_initialization
    likelihoods_simulator = likelihoods_simulator
    new_state_index = new_state_index
    outcome_simulator = outcome_simulator
    