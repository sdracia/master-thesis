import random
import copy
import hvplot.pandas
hvplot.extension('bokeh')
from bokeh.plotting import show
import pandas as pd
import plotly.express as px
import scipy.constants as constants
from sympy.physics.wigner import wigner_3j
import numpy as np


from cah_tools import *

class CaOHModel:
    """ This generates a model for the CaOH molecule
    The model is based on the linear rotor model with nuclear spin 1/2
    The molecule is assumed to be in a magnetic field given by B_0
    The molecule is assumed to be in a laser field with a relative strength given by relative_laser_field
    The molecule is assumed to be in a black body temperature given by temperature
    The model uses the parameters for CaOH from Michał Tomza """

    def __init__(self,
                 B_0=.5*0.357E-3,
                 jmax=30,
                 Clist=[1e10, 0, 0, 0],  # This is just a guess for now. Needs to be updates
                 glist = [-0.036],
                 cijlist = [1.49e3]
                 ):
        """
        Args: B_0 = magnetic field in Tesla, jmax = maximum J value,
        Clist = list of rotational and centrifugal constants, for rotational constant C set [C, 0, 0, 0]
        glist = list of g factors, set to [g] if all g factors are the same
        cijlist = list of cij constants, set to [c] if all cij constants are the same
        """
        self.Clist = Clist
        self.plot_data = None
        if len(glist) == 1:
            self.glist = glist * jmax
        if len(cijlist) == 1:
            self.cijlist = cijlist * jmax
        self.molecule_caoh = [self.glist, self.cijlist, self.Clist]

        self.jmax = jmax
        self.B_0 = B_0
        self.energy_list = []
        self.rabi_list = []
        self.build_index()

    def build_index(self):
        """Builds a list of indices that are related to the quantum numbers for the linear rotor model with nuclear spin 1/2
        """
        counter = 0
        index_list = []
        for j in range(1, self.jmax):
            for i_m in range(2 * j + 1):
                m = i_m - j  #(i,-i-.5,0)
                for i in [-1, 1]:
                    index_list.append([j, m+i/2, i])
        self.index_list = index_list
        self.index_len = len(index_list)


    def init_transition_frequencies(self, delta_m=1):
        """Creates a list of transition frequencies"""
        self.energy_list = []
        for [j,m,i] in self.index_list:
            if i == 0:
                i = -1
            try:
                freq = transition_energy([j, m, i], [j, m + delta_m, i], self.molecule_caoh, self.B_0) / constants.h
            except RuntimeError:
                freq = 0 # Or should we add NaN here?
            self.energy_list.append(freq)
        self.energy_list = np.array(self.energy_list)

    def init_rabi_frequencies(self, relative_laser_field=1e4, delta_m=1, polarization=0):
        """Creates a list of Rabi frequencies
        args: relative_laser_field = relative strength of the laser field,
        args: delta_m = change in m
        args: polarization = polarization of the laser field (currently unused)"""
        self.rabi_list = []
        for [j, m, i] in self.index_list:
            if i == 0:
                i = -1
            transition = [1, 0]
            # rel_rate = transition_coupling([j,m,i], [j+1,m+delta_m,i], transition,
            #                               self.molecule_caoh, self.B_0)
            # transition = [-1, delta_m]
            # rel_rate *= transition_coupling([j, m, i], [j + 1, m + delta_m, i], transition,
            #                                self.molecule_caoh, self.B_0)

            rel_rate = self.coupling(j, m+.5, 1, delta_m)
            rel_rate *= self.coupling(j, m+.5, 1, 0)

            #We are cheating here and use the delta_j=1 coupling rate!
            rabi_rate = rel_rate*relative_laser_field
            self.rabi_list.append(rabi_rate)
        self.rabi_list = np.array(self.rabi_list)

    def excitation_probability(self, duration, rabi_rate, detuning, dephased=False):
        """Calculate the excitation probability for a given duration, Rabi rate and detuning"""
        #print('a',detuning, duration*rabi_rate)
        omega_t = np.sqrt(rabi_rate**2 + detuning**2)
        if dephased:
            exc_prob = abs(rabi_rate / omega_t) ** 2 * .5
        else:
            exc_prob = abs(rabi_rate/omega_t)**2 * np.sin(omega_t*duration*np.pi/2)**2
        return exc_prob

    def init_probabilities(self, param_list=[1e-2, 1.0e3], dephased=False):
        """Creates a list of excitation probabilities for a given list of parameters
        args: param_list = list of parameters [[duration1, detuning1], [duration2, detuning2]]
        """
        self.probability_list = []
        self.param_list = param_list
        for duration, detuning in param_list:
            exc_prob = self.excitation_probability(duration, self.rabi_list, self.energy_list-detuning,
                                                   dephased=dephased)
            self.probability_list.append(exc_prob)

    def coupling(self, j, m, delta_j, delta_m, pol=0):
        """Calculate transition rate in rad/s for dipole interaction between linear rotor states by inputing a rate parameter.
        Args: rate = rate for transition [1/s], j = starting j, m = starting m, delta_j = change in j, delta_m = change in m.
        """
        J0 = j
        J1 = j + delta_j
        m0 = m
        m1 = m + delta_m
        dm = delta_m
        coupling = (
            np.sqrt((2 * J0 + 1) * (2 * J1 + 1))
            * float(wigner_3j(J0, 1, J1, m0, dm, -m1))
            * float(wigner_3j(J0, 1, J1, 0, 0, 0))
            * (-1) ** m1
        )
        return coupling

    def init_pumping(self, param_list):
        """Creates a list of pumping operators for a given list of parameters
        args: param_list = list of parameters [[duration1, detuning1], [duration2, detuning2]]"""
        pumping_list = []
        self.pumping_op_list = []
        target_list = []
        no_pump_list = []
        for idx in range(self.index_len):
            (j,m,i) = self.index_list[idx]
            if j+m-i/2 < 1e-4:
                #print(f'edge state: {j,m,i}')
                target_list.append([idx, idx])
            else:
                target_idx = self.index_list.index([j,m-1,i])
                target_list.append([idx, target_idx])

        #print(f'targets list length {len(target_list)}')

        self.pump_param_list = param_list
        for duration, detuning in param_list:
            exc_prob = self.excitation_probability(duration, self.rabi_list, self.energy_list - detuning, dephased=True)
            pumping_list.append(exc_prob)
            pumping_op_array = np.zeros((self.index_len, self.index_len))
            for idx, p_exc in enumerate(exc_prob):
                try:
                    orig_idx = target_list[idx][0]
                    pump_idx = target_list[idx][1]
                    pumping_op_array[pump_idx, orig_idx] = p_exc
                    pumping_op_array[orig_idx, orig_idx] += 1 - p_exc
                except SyntaxError:
                    print(f'Index error {pump_idx}, {orig_idx}, {self.index_list[pump_idx]}, {self.index_list[orig_idx]}')
                    try:
                        no_pump_idx = no_pump_list[orig_idx]
                        pumping_op_array[no_pump_idx, no_pump_idx] = 1.0
                    except IndexError:
                        pass

            self.pumping_op_list.append(pumping_op_array)

    def plot_distribution(self, probabilities, x_name='detuning', title=None, y_label='Probability'):
        """Plots the distribution of a given list of probabilities"""
        df = self.build_dataframe(probabilities)
        #fig = df.hvplot.scatter(x=x_name, y='probability', by='i', title=title)
        #return fig
        fig = px.scatter(df, x=x_name, y="probability", color='i',
                         custom_data=[df['J'], df['m'], df['i'],df['detuning']],
                         title=title, color_discrete_sequence=px.colors.qualitative.Antique,
                         color_continuous_scale=px.colors.sequential.Bluered,
                         labels={"probability": y_label})
        fig.update_traces(
            hovertemplate='detuning:%{customdata[3]}<br>probability:%{y}<br>J:%{customdata[0]}<br>m:%{customdata[1]}<br>i:%{customdata[2]}')
        fig.show()

    def build_dataframe(self, probabilities):
        """Builds a pandas dataframe for a given list of probabilities as required for plot_distribution"""
        final_array = np.zeros((5, self.index_len))
        final_array[0:3, :] = np.transpose(np.array(self.index_list))
        final_array[3, :] = np.array(self.energy_list)
        final_array[4, :] = np.array(probabilities)
        df = pd.DataFrame(np.transpose(final_array), columns=['J', 'm', 'i', 'detuning', 'probability'])
        return df

    def add_to_plot(self, probabilities, cycle=None):
        """Adds a list of probabilities to the current plot dataframe"""
        if cycle is None:
            if self.plot_data is None:
                cycle = 0
            else:
                cycle = int(np.max(self.plot_data['cycle'])) + 1

        df = self.build_dataframe(probabilities)
        df.insert(1, 'cycle', np.zeros((self.index_len,1))+cycle)
        if self.plot_data is None:
            self.plot_data = df
        else:
            self.plot_data = pd.concat([self.plot_data,df], ignore_index=True)

    def plot_animation(self, x_name='detuning', title=None):
        """Plots an animation of the distribution of probabilities"""
        df = self.plot_data
        fig = px.scatter(df, x=x_name, y="probability", color='i',
                         custom_data=[df['J'], df['m'], df['i'], df['detuning']],
                         animation_frame='cycle',
                         title=title, color_discrete_sequence=px.colors.qualitative.Antique,
                         color_continuous_scale=px.colors.sequential.Bluered)
        fig.update_layout(yaxis_range=(0, 1))
        fig.update_traces(
            hovertemplate='detuning:%{customdata[3]}<br>probability:%{y}<br>J:%{customdata[0]}<br>m:%{customdata[1]}<br>i:%{customdata[2]}')
        fig.show()

    def reset_plot(self):
        """Resets the plot dataframe"""
        self.plot_data = None

    # def plot_distribution(self, probabilities, x_name='detuning', title=None):
    #     df = self.build_dataframe(probabilities)
    #     #fig = df.hvplot.scatter(x=x_name, y='probability', by='i', title=title)
    #     #return fig
    #     fig = px.scatter(df, x=x_name, y="probability", color='i',
    #                      custom_data=[df['J'], df['m'], df['i'],df['detuning']],
    #                      title=title, color_discrete_sequence=px.colors.qualitative.Antique,
    #                      color_continuous_scale=px.colors.sequential.Bluered)
    #     fig.update_traces(
    #         hovertemplate='detuning:%{customdata[3]}<br>probability:%{y}<br>J:%{customdata[0]}<br>m:%{customdata[1]}<br>i:%{customdata[2]}')
    #     fig.show()



class BayesianEstimation:
    """ This class is used to estimate the state of a CaOH molecule using Bayesian estimation
    It uses a CaOHModel object to generate the model for the molecule
    The Bayesian estimation is performed with the method run_estimation
    """
    def __init__(self, model=None, temperature=50.0, **kwargs):
        """
        Args: model = CaOHModel object, temperature = temperature in Kelvin
        KwArgs: kwargs = keyword arguments for CaOHModel
        """
        self.entropy_list = []
        self.outcome_list = []
        self.spectrum_list = []
        self.utility_list = []
        if model is None:
            model = CaOHModel(**kwargs)
            model.init_transition_frequencies()
            model.init_rabi_frequencies()
        self.model = model
        self.temperature = temperature
        self.init_prior()
        #self.init_measurement_setting()

    def apply_pumping(self, pumping_cycles=1):
        """
        Applies a pumping operation to the prior distribution. The default operator from the CaOHModel is used.
        """
        pumped_prob = self.prior
        for idx in range(pumping_cycles):
            for pump_op in self.model.pumping_op_list:
                pumped_prob = np.dot(pump_op, pumped_prob)
            self.prior = pumped_prob

    def init_measurement_setting(self, max_excitation=0.8):
        """
        Creates a list of outccome probabiltities for all measurement settings
        The list includes a resonant measurement setting for each state
        There are probably many overlapping detunings
        """
        N0 = self.model.index_len
        param_list = []
        for idx, (j,m,i) in enumerate(self.model.index_list):
            if (j + m - i / 2 < 1e-4) and (i==-1):
                detuning = self.model.energy_list[idx]
                duration = max_excitation /self.model.rabi_list[idx] #1/max_rabi_freq
                print(f'edge state {j, m, i} : detuning: {detuning}')
                if not np.isnan(duration):
                    param_list.append([duration, detuning])
                else:
                    print(f'NaN duration {idx}')
        self.model.init_probabilities(param_list)

    def update_distribution(self):
        """
        Updates the prior distribution based on a simulated measurement outcome
        """
        self.guess_idx = self.get_next_setting()
        lh0, lh1, p0 = self.get_measurement(self.guess_idx)
        this_rand = np.random.rand()
        # print(p1, this_rand)
        if this_rand < p0:
            outcome = 0
            posterior = self.prior * lh0
        else:
            outcome = 1
            posterior = self.prior * lh1
        self.posterior = posterior / np.sum(posterior)
        self.outcome = outcome
        self.prob_0 = p0

    def calc_entropy(self, prob, prob1=1.0):
        """Calculate the Entropy (for prob1=1.0) or the Kulback-Liebler divergence
        See: https://pages.nist.gov/optbayesexpt/manual.html#philosophy-and-goals"""
        entropy = np.sum(prob*np.log(prob/prob1), where=(prob1 != 0) | (prob != 0))
        return entropy

    def calc_utility(self, data=None):
        """Calculates the utility for the current prior for each measurement setting"""
        if data is None:
            data = self.prior
        utility_list = []
        for meas_idx, meas_prob in enumerate(self.model.probability_list):
            lhood_0, lhood_1, probability_0 = self.get_measurement(meas_idx, data=data)
            entropy_0 = self.calc_entropy(lhood_0*data+1e-5, data+1e-5)
            entropy_1 = self.calc_entropy(lhood_1*data+1e-5, data+1e-5)
            utility = entropy_0*probability_0 + entropy_1*(1-probability_0)
            utility_list.append(utility)
        return utility_list

    def calc_spectrum(self, data=None, duration=1e3, detuning_list=np.linspace(-5000, 25000, 50)):
        """Calculates the spectrum for a given duration and detuning list given the current prior distribution"""
        if data is None:
            data = self.prior
        model = copy.deepcopy(self.model)
        param_list = []
        for detuning in detuning_list:
            param_list.append([duration, detuning])
        model.init_probabilities(param_list)
        prob_list = []
        for probabilities in model.probability_list:
            prob_list.append(np.sum(probabilities * data))
        return detuning_list, prob_list

    def get_measurement(self, measurement_idx, data=None):
        """Calculates the likelihood distributions for both outcomes and the probability for outcome 0 for a given measurement setting"""
        if data is None:
            data = self.prior
        prob_array = self.model.probability_list[measurement_idx]
        lhood_0 = prob_array
        lhood_1 = 1-prob_array
        probability_0 = np.sum(data * lhood_0)
        return lhood_0, lhood_1, probability_0

    def get_next_setting(self):
        """Returns the next measurement setting index that maximizes the utility"""
        util_list = self.calc_utility()
        sorted_list = np.argsort(np.array(util_list))
        idx = random.choice(sorted_list[0:5])
        #idx = np.argmin(util_list)
        #print(f'Optimum measurment index: {idx}; detuning: {self.model.param_list[idx]}' )
        return int(idx)
        #return random.randint(0, len(self.model.probability_list)-1)
    def init_prior(self):
        """Initializes the prior distribution based on the temperature of the system"""
        T = self.temperature
        rotational_const = self.model.Clist[0]
        h = constants.h
        kb = constants.k
        p_list = np.zeros(self.model.index_len)
        for idx, (J,m,i) in enumerate(self.model.index_list):
            p_list[idx] = np.exp(-h * rotational_const * J * (J + 1) / (kb * T))
        self.prior = p_list / np.sum(p_list)

    def init_edge_prior(self):
        """Initializes the prior distribution based on the temperature of the system assuming only the edge states are populated"""
        self.init_prior()
        for idx in range(self.model.index_len):
            (j, m, i) = self.model.index_list[idx]
            if (j + m - i / 2 < 1e-4) and i==-1:
                pass
            else:
                self.prior[idx] = 0
        self.prior = self.prior / np.sum(self.prior)


    def run_estimation(self, no_updates=5, save_data=False, apply_pumping=False):
        """Runs the Bayesian estimation for a given number of updates"""
        for i in range(no_updates):
            if apply_pumping:
                self.apply_pumping()
            self.update_distribution()
            print(self.guess_idx, self.model.param_list[self.guess_idx], self.prob_0, self.outcome)
            self.prior = self.posterior
            if save_data:
                self.model.add_to_plot(self.prior)
                self.spectrum_list.append(self.calc_spectrum())
                self.utility_list.append(self.calc_utility())
            self.outcome_list.append([self.guess_idx, self.model.param_list[self.guess_idx], self.prob_0, self.outcome])
            self.entropy_list.append(self.calc_entropy(self.prior))


if __name__ == '__main__':
    B = BayesianEstimation(temperature=30, jmax=15)
    B.model.init_rabi_frequencies(relative_laser_field=1e4)
    B.init_measurement_setting()
    #for (duration, detuning), probs in zip(B.model.param_list, B.model.probability_list):
    #    B.model.plot_distribution(probs, x_name='J', title=f'Probability {detuning}')
    B.model.plot_distribution(B.prior, x_name='J',title='Initial distribution')
    B.model.init_pumping(param_list=[[1e3, 300]])
    B.apply_pumping(pumping_cycles=10)
    B.model.plot_distribution(B.prior, x_name='J', title=f'after pumping')
    B.model.reset_plot()
    for idx in range(100):
        B.run_estimation(no_updates=1)
        B.model.add_to_plot(B.prior, idx)

    #print(B.model.plot_data)
    B.model.plot_animation(x_name='J', title="Bayesian")
    #B.model.plot_distribution(B.prior, x_name='J', title=f'after Bayesian estimation, cycle {idx}, J_guess: {B.model.index_list[B.guess_idx]}, outcome: {B.outcome}')
