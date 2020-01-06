#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file presents examples of how to use the class NNMIP from file nn_mip.py
"""

# Libs
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from docplex.mp.solution import SolveSolution
# documentation
# http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html

# Own Modules
from source.nn_mip import NNMIP

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'

# %% Example 1 Toy  for 2-dim input, i.e. 2 items
# %% Define two NNs

# NN1
input1 = Input(shape=(2, ))
x = Dense(3, activation='relu')(input1)
x = Dense(3, activation='relu')(x)
predictions1 = Dense(1, activation='relu')(x)
KM1 = Model(inputs=input1, outputs=predictions1)
# NN2
input2 = Input(shape=(2,))
x = Dense(3, activation='relu')(input2)
x = Dense(3, activation='relu')(x)
predictions2 = Dense(1, activation='relu')(x)
KM2 = Model(inputs=input2, outputs=predictions2)

# Set the weights for these NNs.
# NN1
W1 = np.array([[1, 0, 0], [0, 1, 0]])
b1 = np.array([0, 0, 1])
W2 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
b2 = np.array([2, 2, 2])
W3 = np.array([[5], [5], [5]])
b3 = np.array([10])
KM1.set_weights([W1, b1, W2, b2, W3, b3])
# NN2
W1 = np.array([[1, 0, 0], [0, 1, 0]])
b1 = np.array([0, 0, 1])
W2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b2 = np.array([2, 2, 2])
W3 = np.array([[5], [5], [5]])
b3 = np.array([10])
KM2.set_weights([W1, b1, W2, b2, W3, b3])

# Collect them in a dict.
NNs = {'Bidder_0': KM1, 'Bidder_1': KM2}

# check weights and output for given input.
KM1.get_weights()
print(KM1.predict(np.array([[0, 0]])))
print(KM1.predict(np.array([[1, 0]])))
print(KM1.predict(np.array([[0, 1]])))
print(KM1.predict(np.array([[1, 1]])))
KM2.get_weights()
print(KM2.predict(np.array([[0, 0]])))
print(KM2.predict(np.array([[1, 0]])))
print(KM2.predict(np.array([[0, 1]])))
print(KM2.predict(np.array([[1, 1]])))
# %%
# (1) without tightening bounds
X = NNMIP(models=NNs, L=1000)  # initialize NN_MIP instances
X.initialize_mip(verbose=True)  # initialize Mip attribute in this class
s = X.solve_mip(log_output=True, time_limit=None, mip_start=None)  # solve Mip
print(s)  # print solution
X.soltime  # timings
X.print_optimal_allocation()  # print only optimal allocation
# print constraints
for c in X.Mip.iter_constraints():
    print(c)
X  # check echoe on python shell

# (2) with tightening bounds IA
del X
X = NNMIP(NNs, L=1000)
X.tighten_bounds_IA(upper_bound_input=[1, 1])  # tighten bounds with IA
X.print_upper_bounds()  # print upper bounds
X.initialize_mip(verbose=True)
s = X.solve_mip(log_output=True, time_limit=None, mip_start=None)
X.soltime
X.print_optimal_allocation()
for c in X.Mip.iter_constraints():
    print(c)
X

# (3) with tightening bounds LP
del X
X = NNMIP(NNs, L=1000)
X.tighten_bounds_LP(upper_bound_input=[1, 1])  # tighten bounds with LP
X.print_upper_bounds()  # print upper bounds
X.initialize_mip(verbose=True)
s = X.solve_mip(log_output=True, time_limit=None, mip_start=None)
X.soltime
X.print_optimal_allocation()
for c in X.Mip.iter_constraints():
    print(c)
X

# (4) re-solve with a warm start
start = s.as_dict()
X = NNMIP(NNs, L=1000)
X.initialize_mip(verbose=True)
X.solve_mip(mip_start=SolveSolution(X.Mip, start))
X.x_star
X.print_optimal_allocation()
X
