# import qutip as q
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import copy
import scipy.optimize as opt
from scipy import constants
from sympy.physics.wigner import wigner_3j
from sympy import *
from functools import reduce
import operator
# from rot_qec_tools import *

def energy_sublevels(state,molecule,B):
    '''Function for computing energy sublevel eigenvalues of a linear rotor with nuclear spin 1/2 such as CaH+. Args: state = [J,m,xi], molecule = [g,cij,Clist], B = external magnetic field.'''
    J = state[0]
    m = state[1]
    xi = state[2]
    g = molecule[0][0]
    cij = molecule[1][0]
    #if abs(m) > J:
    #    raise RuntimeError(f'State ({J},{m}) not available')
    X = (1/2) * np.sqrt(constants.h**2 * cij**2 * ((J + (1/2))**2 - m**2) + (constants.h * cij * m - constants.physical_constants['nuclear magneton'][0] * B * (g - constants.physical_constants['proton g factor'][0]))**2)
    Y = (-1/2) * (constants.physical_constants['nuclear magneton'][0] * B * (g - constants.physical_constants['proton g factor'][0]) - m * constants.h * cij)
    if m == - J - 1/2 or m == J + 1/2:
        nrg = - xi * (g * J + constants.physical_constants['proton g factor'][0] / 2) * constants.physical_constants['nuclear magneton'][0] * B - constants.h * cij * J / 2
    else:
        nrg = constants.h * cij / 4 - constants.physical_constants['nuclear magneton'][0] * B * g * m - (xi) * X
    return [nrg, X, Y]

def energy_centroids(state,molecule):
    '''Function for computing energy centroid eigenvalues of a linear rotor given molecular constants for centrifugal corrections. Args: state = [J,m,xi], molecule = [g,cij,Clist].'''
    Clist = molecule[2]
    J = state[0]
    kmax = len(Clist)
    nrg = constants.h * np.sum([Clist[k - 1] * J**k * (J + 1)**k for k in range(1,kmax + 1)])
    return nrg

def transition_energy(state_1,state_2,molecule,B):
    '''Function for computing the difference frequency between molecular states. Args: state = [J,m,xi], molecule = [g,cij,Clist], B = external magnetic field.'''
    nrg_2 = energy_centroids(state_2,molecule) + energy_sublevels(state_2,molecule,B)[0]
    nrg_1 = energy_centroids(state_1,molecule) + energy_sublevels(state_1,molecule,B)[0]
    delta_nrg = nrg_2 - nrg_1
    return delta_nrg

