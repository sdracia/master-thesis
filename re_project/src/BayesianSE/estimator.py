from molecules.molecule import CaH, CaOH
from QLS.state_dist import States



from .estimator_utils import init_prior, stop_condition, cross_entropy
from .estimator_likelihood import likelihoods_estimator
from .estimator_update import update_distibution, get_next_setting
from .estimator_measurements import measurement_setting
from .estimator_pumping import reset_prior_pumping, within_run_pumping




class BayesianStateEstimation:

    def __init__(self, 
                 model = None, 
                 temperature = 300, 
                 b_field_gauss = 3.27, 
                 j_max = 50,
                 false_positive_rate = 0.00,
                 false_negative_rate = 0.00):   #####
        
        self.b_field_gauss = b_field_gauss
        self.j_max = j_max

        if model is None:
            model = CaOH.create_molecule_data(b_field_gauss=self.b_field_gauss, j_max=self.j_max)
        
        self.model = model

        self.fpr = false_positive_rate
        self.fnr = false_negative_rate

        # - If model = None, this if condition will be verified (since no "state_dist" is present)
        # - I pass a model to the class: if qls.States have been previously called, and/or optical pumping already applied, 
        # this if condition will not be verified. If qls.States has not been called to the model before, the if condition will be  
        # verified
        if "state_dist" not in self.model.state_df.columns:
            states1 = States(molecule = self.model, temperature = temperature)

        self.temperature = temperature
        
        self.init_prior()
        self.history_list = []

        # If we pass to the class a model that has already been pumped
        self.after_pumping_dist = self.model.state_df["state_dist"].copy()

    init_prior = init_prior
    reset_prior_pumping = reset_prior_pumping
    within_run_pumping = within_run_pumping
    likelihoods_estimator = likelihoods_estimator
    measurement_setting = measurement_setting
    update_distibution = update_distibution
    get_next_setting = get_next_setting
    stop_condition = stop_condition
    cross_entropy = cross_entropy

