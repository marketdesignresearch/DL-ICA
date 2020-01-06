#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file presents examples of how to use the class Economies from file economies.py
"""

# Libs
import logging
# Own Modules
from source.economies import Economies
from source.pysats import PySats

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'
# %% define logger
# clear existing logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# LOG DEBUG TO CONSOLE
logging.basicConfig(level=logging.DEBUG, format='%(message)s', filemode='w')
# %% create SATS value model instance
value_model = PySats.getInstance().create_lsvm(seed=1)
# %% Create DNN parameters
# (1) Value model parameters
model_name = 'LSVM'
N = 6  # number of bidders
M = 18  # number of items
bidder_types = 2
bidder_ids = list(range(0, 6))
print('\nValue Model:', model_name)
print('Number of Bidders: ', N)
print('Number of BidderTypes: ', bidder_types)
print('Number of Items: ', M)
scaler = False
# (2) Neural Network Parameters
epochs = 512
batch_size = 32

# national bidder LSVM: id=0, GSVM:id=6, MRVM:id=7,8,9
regularization_N = 0.00001
learning_rate_N = 0.005
layer_N = [10, 10, 10]
dropout_N = True
dropout_prob_N = 0.05

# regional bidders LSVM: id=1-5, GSVM:id=0-5, MRVM:id=3,4,5,6
regularization_R = 0.00001
learning_rate_R = 0.01
layer_R = [16, 16]
dropout_R = True
dropout_prob_R = 0.05

# local bidders MRVM:id=0,1,2
regularization_L = 0.00001
learning_rate_L = 0.01
layer_L = [10, 10, 10]
dropout_L = True
dropout_prob_L = 0.05

parameters = {}
if model_name == 'LSVM':
    for bidder_id in bidder_ids:
        if bidder_id == 0:
            parameters['Bidder_{}'.format(bidder_id)] = (regularization_N, learning_rate_N, layer_N, dropout_N, dropout_prob_N)
        else:
            parameters['Bidder_{}'.format(bidder_id)] = (regularization_R, learning_rate_R, layer_R, dropout_R, dropout_prob_R)
if model_name == 'GSVM':
    for bidder_id in bidder_ids:
        if bidder_id == 6:
            parameters['Bidder_{}'.format(bidder_id)] = (regularization_N, learning_rate_N, layer_N, dropout_N, dropout_prob_N)
        else:
            parameters['Bidder_{}'.format(bidder_id)] = (regularization_R, learning_rate_R, layer_R, dropout_R, dropout_prob_R)
if model_name == 'MRVM':
    for bidder_id in bidder_ids:
        if bidder_id in [0, 1, 2]:
            parameters['Bidder_{}'.format(bidder_id)] = (regularization_L, learning_rate_L, layer_L, dropout_L, dropout_prob_L)
        if bidder_id in [3, 4, 5, 6]:
            parameters['Bidder_{}'.format(bidder_id)] = (regularization_R, learning_rate_R, layer_R, dropout_R, dropout_prob_R)
        if bidder_id in [7, 8, 9]:
            parameters['Bidder_{}'.format(bidder_id)] = (regularization_N, learning_rate_N, layer_N, dropout_N, dropout_prob_N)
# %%
E = Economies(value_model=value_model, c0=10, ce=5, min_iteration=1, epochs=epochs, batch_size=batch_size, regularization_type='l2', L=5000, Mip_bounds_tightening='IA', warm_start=True, scaler=None)  # create instance of the class Economies
E.set_NN_parameters(parameters=parameters)  # set DNN parameters
# check initialized attributes
E.value_model
E.bidder_ids
E.N
E.good_ids
E.M
E.c0
E.ce
E.batch_size
E.epochs
E.L
E.regularization_type
E.Mip_bounds_tightening
E.warm_start
E.scaler
E.pvm_allocation
E.min_iteration
E.economies  # orderedDict containing all economies and the corresponding bidder ids of the bidders which are active in these economies.
E.economies_names  # orderedDict containing all economies and the corresponding bidder names (as strings) of the bidders which are active in these economies.
E.sample_weights_on
E.weight_scaling

# dynamic attributes per economy
E.elicited_bids
E.iteration
E.status
E.WDP_allocations
E.fitted_scaler
E.elapsed_time_mip
E.warm_start_sol
# dynamic attributes per bidder
E.payments
E.total_bounds
E.actual_queries_elicitation
# dynamic attributes per economy & bidder
E.bounds
E.weights
E.argmax_allocation  # two dimensional list per bidder and economy: [bundle, value for bundle]
E.NN_parameters
E.NN_models
E.losses
# %%

E.set_initial_bids(seeds_random_bids=None)  # create c0 inital bids V^0, uniformly sampled at random, for each bidder. (null bundle is added addtionally with value 0)
E.elicited_bids  # look them up
parameters  # look them up
E.set_NN_parameters(parameters=parameters)   # set DNN parameters
sample_weight_on = False
sample_weight_scaling = [30, 10, 10, 10, 10, 10]  # (not used in AAAI 2020 paper) Scaling vector for errors for the 6 bidders in LSVM. Since the MAE error for the national bidder is usually larger the corretion weight for the this datapoint in for next training of DNNs is MAE/30. For the regional bidders it is MAE/10.
# activate weights (not used in AAAI 2020 paper)
if sample_weight_on:
    E.activate_weights(sample_weight_scaling=sample_weight_scaling)
# do one round of the PEA for a specific economy
economy_key = 'Marginal Economy -{3}'  # select economy
E.info(economy_key=economy_key)
E.status  # check status
E.calculate_argmax_allocation(economy_key=economy_key)  # Estimation & Optimization Step
E.NN_models  # check trained DNNs
E.argmax_allocation  # check argmax allocation, not yet filled with values in the second entries of the lists.
# E.do_query(economy_key=economy_key, allocation=E.argmax_allocation)  # version 1 with total caps
E.do_query_2(economy_key=economy_key, allocation=E.argmax_allocation)  # version 2 with individual caps: AAAI 2020 paper
E.total_bounds
E.bounds
E.elicited_bids[economy_key]  # check elicited bids again which argmax bundles were added
# E.update_main_economy()  # iteratively update main economy, not used in AAAI 2020 paper
E.reset_keras_models()  # reset models
E.NN_models
E.reset_argmax_allocations()  # reset argmax allocations
E.argmax_allocation
# E.add_all_bids_to_main_economy()  # add all bids to main economy, not used in AAAI 2020 paper
E.info(economy_key=economy_key)
