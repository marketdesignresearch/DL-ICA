#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file implements the class Economies. This class is used for simulating the DNN-based ICA, i.e., the PVM mechanism (Algorithm 2 in Section 2), based on DNNs (Deep Neural Networks).
Specifically, this class stores all the information about the economies in the iterative procedure of the DNN-based preference elicitation algorithm (Algorithm 1 in Section 2).

Economies has the following functionalities:
    0.CONSTRUCTOR: __init__(self, value_model, c0, ce, min_iteration, epochs, batch_size, regularization_type, L, Mip_bounds_tightening, warm_start, scaler)
        value_model = auction instance created with SATS (GSVM, LSVM, MRVM)
        c0 = number of intial data points, i.e., bundle-value pairs (queried randomly per bidder prior to the elicitation procedure) (positive integer)
        ce = maximal number of possible value queries in the preference elicitation algorithm (PEA) per bidder (positive integer)
        min_iteration = minimial iterations for a PEA run: set equal to 1 in the paper, hence no restriction and original version of PVM
        epochs = epochs in training the DNNs
        batch_size = batch size in training the deep neural networks (DNNs)
        regularization_type = regulairzation type for the affine mappings between layers: 'l1', 'l2', 'l1_l2' (string)
        L = global big-M constraint for the MIPs
        Mip_bounds_tightening = bounds tightening procedure: False, IA (Intervall Arithmetic), LP (Linear Program Relaxations)
        warm_start = boolean, if MIP solution of prior auction round is used as a warm start for the MIPs
        scaler = scaler is a instance from sklearn.MinMaxScaler() used for scaling the values of the bundle-value pairs
        This method creates a instance of the class Economies. Details about the attributes can be found below.
    1.METHOD: info(self, economy_key=None)
        economy_key = key for a specific economy, e.g., 'Marginal Economy -{3}': bidder 3 excluded (string)
        This method reutnr information of either a specific economy if economy_key is specified or for all economies.
    2.METHOD: activate_weights(self, sample_weight_scaling)
        sample_weight_scaling = boolean, if elicited bundle-value pairs should be weighted (more) the next time the DNNs are trained.
        This method activates the weights and sets the initial weightvector.
    3.METHOD: set_initial_bids(self, seeds_random_bids)
        seeds_random_bids = seed used for random sampling of the initial bids
        This method creates sets of initial bundle-value pairs, sets the elicited bids attribute and the scaler attribute, i.e., bids.
        It calls the function util_github.initial_bids_pvm_UNIF (uniformly at random from the bundle space 2^m) or util_github.initial_bids_pvm (SATS sampling via Normal Distribution see 4.METHOD in Gsvm.py for example.)
    4.METHOD: reset_argmax_allocations(self)
        This method cleares the argmax allocation a^(t) from the round t.
    5.METHOD: reset_weights(self, economy_key)
        economy_key = key for a specific economy, e.g., 'Main Economy': no bidder excluded (string)
        This method resets the weightvector for a economy specified by economy_key used for weighting (more or less) elicited bundle-value pairs when fitting the DNNs.
    6.METHOD: reset_keras_models(self)
        This method resets the keras models, i.e., the DNNs from the previous round.
    7.METHOD: update_bids(self, economy_key, bidder_id, bundle_to_add, value_to_add)
        economy_key = key for a specific economy, e.g., 'Marginal Economy -{0}':bidder 0 excluded (string)
        bidder_id = unique bidder id (positive integer)
        bundle_to_add = m dimensional indicator vector representing the bundle of items (np.array)
        value_to_add = value of bidder bidder_id to the corresponding bundle_to_add
        This method adds a bundle-value pair for bidder bidder_id in economy economy_key to the corresponding set of elicited bids
    8.METHOD: check_bundle_contained(self, argmax_bundle, bidder_id)
        argmax_bundle = m dimensional vector representing a^(t)_i where i= bidder_id
        bidder_id = unique bidder id (positive integer)
        This method checks if the bundle already has been queried from bidder bidder_id in ANY economy.
    9.METHOD: update_weights(self, economy_key, bidder_id, weights)
        economy_key = key for a specific economy, e.g., 'Marginal Economy -{4}':bidder 4 excluded (string)
        bidder_id = unique bidder id (positive integer)
        weights = new weights vector
        This method updates the weightvector self.weights[economy_key][bidder_id] for an economy sepcified by economy_key and a bidder specified as bidder_id.
    10.METHOD: update_main_economy(self)
        This method updates the main economy in each iteration by adding all the elicited bundles from the marginal economies after a single iteration of the PEA to the main economy.
    11.METHOD: add_all_bids_to_main_economy(self)
        This method adds at the end of all PEAs (in contrast to update_main_economy()) all the elicited bundles from the marginal economies to the main economy.
    12.METHOD: _fit_NNs(self, Data, parameters, economy_key, sample_weights=None)
        Data = current set of elicited bundle-value pairs in format [[[X^(0)_train,Y_train^(0)],...,[X^(N)_train,Y_train^(N)]], scaler]
        parameters = DNNs parameters
        economy_key = key for a specific economy, e.g., 'Marginal Economy -{2}':bidder 2 excluded (string)
        sample_weights = weightvector for data points
        This method fits all DNNs to a dataset of bundle-value pairs specified by Data. It return a OrderedDict of fitted keras models.
    13.METHOD: calculate_argmax_allocation(self, economy_key)
        economy_key = key for a specific economy, e.g., 'Marginal Economy -{2}':bidder 2 excluded (string)
        This method calculates the argmax allocation a^(t) for an economy specified by economy_key and sets self.argmax_allocation[economy_key] accordingly.
    14.METHOD: do_query(self, economy_key, allocation):
        economy_key = key for a specific economy, e.g., 'Marginal Economy -{2}':bidder 2 excluded (string)
        allocation = an argmax allocation a^(t) calculated via the method calculate_argmax_allocation.
        This method queries each bidder i the bundle a^(t)_i in the argmax alocation in the PEA.
        NEW VERSION WITH TOTAL CAPS PER WHOLE PVM RUN AND NEW ALLOCATION RULE ("min_iteration", "reweighting" and "add marginals" can be specified optionally)
    15.METHOD: do_query_2(self, economy_key, allocation):
        economy_key = key for a specific economy, e.g., 'Marginal Economy -{2}':bidder 2 excluded (string)
        allocation = an argmax allocation a^(t) calculated via the method calculate_argmax_allocation.
        This method queries each bidder i the bundle a^(t)_i in the argmax alocation in the PEA.
        AAAI 2020 PAPER VERSION WITH INDIVIDUAL CAPS PER ELICITATION RUN (changes to version do_query marked with ***)
    16.METHOD: solve_WDP(self, economy_key=None)
        economy_key = key for a specific economy, e.g., 'Marginal Economy -{2}':bidder 2 excluded (string)
        This method solves the WDP and sets WDP_allocation for an economy specified by economy_key based on the elicited bids in this economy after the PEA has finished. Happens at the end of PVM after the PEA for each economy has stopped.
    17.METHOD: calculate_finalpvm_alloc(self)
        This method calculates the PVM allocation based on all WDP_alocations calculated via solve_WDP(). (see Algorithm 2 in Section 2)
    18.METHOD: calculate_payments(self)
        This method calculates the VCG-style payments. (see Algorithm 2 in Section 2)

See Test_Class_Economies_github.py for an example of how to use the class Economies.
"""

# Libs
import itertools
import sys
import util_github
import numpy as np
import time
import re
from keras import backend as K
from collections import OrderedDict
import logging
# CPLEX: Here, DOcplex is used for solving the deep neural network-based Winner Determination Problem.
import docplex
# documentation
# http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html

# Own Modules
from Class_NN_github import NN
from Class_NN_MIP_github import NN_MIP
from Class_WDP_github import WDP


__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'
# %%

# %% MAIN PATH FOR UNSOLVED MIP EXPORTs
PATH = '/home/user/weissteiner/PVM/unsolved_Mips'
# %%


class Economies:

    def __init__(self, value_model, c0, ce, min_iteration, epochs, batch_size, regularization_type, L, Mip_bounds_tightening, warm_start, scaler):

        # static attributes
        self.value_model = value_model  # auction instance from SATS: LSVM, GSVM or MRVM generated via PySats.py.
        self.bidder_ids = set(value_model.get_bidder_ids())  # bidder ids in this auction instance.
        self.N = len(self.bidder_ids)  # number of bidders
        self.good_ids = set(value_model.get_good_ids())  # good ids in this auction instance
        self.M = len(self.good_ids)  # number of items
        self.c0 = c0  # number of intial data points, i.e., bundle-value pairs (queried randomly per bidder prior to the elicitation procedure)
        self.ce = ce  # maximal number of possible value queries in the preference elicitation algorithm (PEA) per bidder
        self.batch_size = batch_size  # batch size in training the deep neural networks (DNNs)
        self.epochs = epochs   # epochs in training the DNNs
        self.L = L  # global big-M constraint for the MIPs
        self.regularization_type = regularization_type  # regularization type for the affine mappings between layers: 'l1', 'l2', 'l1_l2'
        self.Mip_bounds_tightening = Mip_bounds_tightening  # bounds tightening procedure: False, IA (Intervall Arithmetic), LP (Linear Program Relaxations)
        self.warm_start = warm_start  # boolean, if MIP solution of prior auction round is used as a warm start for the MIPs
        self.scaler = scaler  # scaler is a instance from sklearn.MinMaxScaler() used for scaling the values of the bundle-value pairs
        self.pvm_allocation = None  # final PVM allocation
        self.min_iteration = min_iteration  # minimial iterations for a PEA run: set equal to 1 in the paper, hence no restriction and original version of PVM
        # orderedDict containing all economies and the corresponding bidder ids of the bidders which are active in these economies.
        subsets = list(map(set, itertools.combinations(self.bidder_ids, self.N-1)))
        self.economies = OrderedDict(list(('Marginal Economy -{}'.format(i), econ) for econ, i in zip(subsets, [self.bidder_ids-x for x in subsets])))
        self.economies['Main Economy'] = self.bidder_ids
        # orderedDict containing all economies and the corresponding bidder names (as strings) of the bidders which are active in these economies.
        self.economies_names = OrderedDict(list((key, ['Bidder_{}'.format(s) for s in value]) for key, value in self.economies.items()))
        self.sample_weights_on = False  # boolean, set to True if one wants to use a weight for the last elicited bids in the training of the DNNs in the next round
        self.weight_scaling = OrderedDict(list((key, OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in value))) for key, value in self.economies.items()))  # a weightvector for each bidder that is multiplied to the error of the last elicited bundle, only used if sample_weights_on True

        # dynamic attributes per economy
        self.elicited_bids = OrderedDict(list((key, None) for key, value in self.economies.items()))  # list of elicited bids for each bidder per economy
        self.iteration = OrderedDict(list((key, 1) for key, value in self.economies.items()))  # number of iterations per economy of the PEA
        self.status = OrderedDict(list((key, True) for key, value in self.economies.items()))  # boolean, status of economy: if PEA has stopped.
        self.WDP_allocations = OrderedDict(list((key, None) for key, value in self.economies.items()))  # Allocation of the WDP based on the elicited bids calculated at the end of PVM after the PEA of every economy stopped
        self.fitted_scaler = OrderedDict(list((key, None) for key, value in self.economies.items()))   # scaler object per economy
        self.elapsed_time_mip = OrderedDict(list((key, []) for key, value in self.economies.items()))  # stored MIP solving times per economy
        self.warm_start_sol = OrderedDict(list((key, None) for key, value in self.economies.items()))  # MIP SolveSolution object used as warm start per economy

        # dynamic attributes per bidder
        self.payments = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))  # VCG-style payments in PVM, calulated at the end after WDP_allocations
        self.total_bounds = OrderedDict(list(('Bidder_{}'.format(bidder_id), 0) for bidder_id in self.bidder_ids)) # Total bounds of bidders' value queries across all economies. Used for stopping criteria: <= self.ce.
        self.actual_queries_elicitation = OrderedDict(list(('Bidder_{}'.format(bidder_id), 0) for bidder_id in self.bidder_ids))  # number of actual value queries, that takes into account if the same bundle was queried from a bidder in two different economies that it is not counted twice

        # dynamic attributes per economy & bidder
        self.bounds = OrderedDict(list((key, OrderedDict(list((bidder_id, 0) for bidder_id in value))) for key, value in self.economies_names.items()))  # same as total bounds, but this time individual bounds per economy and per bidder.
        self.weights = OrderedDict(list((key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in self.economies_names.items()))  # total weightvector = weight_scaling*error, per bidder and per economy
        self.argmax_allocation = OrderedDict(list((key, OrderedDict(list((bidder_id, [None, None]) for bidder_id in value))) for key, value in self.economies_names.items()))  # a^(t)_i, argmax bundles per bidder and economy in the PEA
        self.NN_parameters = OrderedDict(list((key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in self.economies_names.items()))  # DNNs parameters as in the Class NN described.
        self.NN_models = OrderedDict(list((key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in self.economies_names.items()))  # Keras models
        self.losses = OrderedDict(list((key, OrderedDict(list((bidder_id, []) for bidder_id in value))) for key, value in self.economies_names.items()))  # Storage for the MAE loss during training of the DNNs

    def info(self, economy_key=None):
        if economy_key is None:
            for economy_key in list(self.economies_names.keys()):
                logging.debug('Economy: %s', economy_key)
                logging.debug('Status: %s', self.status[economy_key])
                logging.debug('Iteration: %s', self.iteration[economy_key])
                logging.debug('Bounds of Economy: %s',  '  '.join("{}:{}".format(k, v) for k, v in self.bounds[economy_key].items()))
                logging.debug('Elicited Bids: %s', [self.elicited_bids[economy_key][0][bidder_id][0].shape[0]-1 for bidder_id in self.economies_names[economy_key]])  # -1 because of added zero bundle
                logging.debug('\n')
            logging.debug('Actual Queries Economies: %s',  '  '.join("{}:{}".format(k, v) for k, v in self.actual_queries_elicitation.items()))
            logging.debug('Total Bounds Economies: %s',   '  '.join("{}:{}".format(k, v) for k, v in self.total_bounds.items()))
            logging.debug('\n')
        else:
            logging.debug('Economy: %s', economy_key)
            logging.debug('Status: %s', self.status[economy_key])
            logging.debug('Iteration: %s', self.iteration[economy_key])
            logging.debug('Elicited Bids: %s', [self.elicited_bids[economy_key][0][bidder_id][0].shape[0]-1 for bidder_id in self.economies_names[economy_key]])  # -1 because of added zero bundle
            logging.debug('Bounds of Economy : %s',  '  '.join("{}: {}".format(k, v) for k, v in self.bounds[economy_key].items()))
            logging.debug('Actual Queries economies: %s',  '  '.join("{}:{}".format(k, v) for k, v in self.actual_queries_elicitation.items()))
            logging.debug('Total Bounds economies: %s',  '  '.join("{}: {}".format(k, v) for k, v in self.total_bounds.items()))
            logging.debug('\n')

    def activate_weights(self, sample_weight_scaling):
        if sample_weight_scaling is not None:
            self.sample_weights_on = True
            self.weight_scaling = OrderedDict(list((key, OrderedDict(list(('Bidder_{}'.format(bidder_id), sample_weight_scaling[bidder_id]) for bidder_id in value))) for key, value in self.economies.items()))
            self.weights = OrderedDict(list((key1, OrderedDict(list((key2, np.ones(self.c0+1)) for key2 in list(self.weights[key1].keys())))) for key1 in list(self.economies.keys())))

    def set_NN_parameters(self, parameters):
        self.NN_parameters = OrderedDict(list((key1, OrderedDict(list((key2, parameters[key2]) for key2 in list(self.NN_parameters[key1].keys())))) for key1 in list(self.economies.keys())))

    def set_initial_bids(self, seeds_random_bids):
        for key, value in self.economies.items():
            logging.debug(key)
            self.elicited_bids[key] = util_github.initial_bids_pvm_UNIF(value_model=self.value_model, c0=self.c0, bidder_ids=value, scaler=self.scaler)  # TRUE UNIFORM SAMPLING
            # self.elicited_bids[key] = util_github.initial_bids_pvm(value_model=self.value_model, c0=self.c0, bidder_ids=value, scaler=self.scaler, seed=seeds_random_bids)  # SATS RANDOM SAMPLING VIA NORMAL DISTRIBUTION
            self.fitted_scaler[key] = self.elicited_bids[key][1]

    def reset_argmax_allocations(self):
        self.argmax_allocation = OrderedDict(list((key, OrderedDict(list((bidder_id, [None, None]) for bidder_id in value))) for key, value in self.economies_names.items()))

    def reset_weights(self, economy_key):
        for bidder_id in self.economies_names[economy_key]:
            dim = self.elicited_bids[economy_key][0][bidder_id][1].shape[0]
            self.weights[economy_key][bidder_id] = np.ones(dim)

    def reset_keras_models(self):
        delattr(self, 'NN_models')
        self.NN_models = OrderedDict(list((key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in self.economies_names.items()))
        K.clear_session()

    def update_bids(self, economy_key, bidder_id, bundle_to_add, value_to_add):
        D = self.elicited_bids[economy_key]
        if any(np.equal(D[0][bidder_id][0], bundle_to_add).all(1)):
            logging.debug('Bundle already elicited')
            del D
            return(False)
        if bundle_to_add is None or value_to_add is None:
            logging.debug('No valid bundle -> cannot add')
            del D
            return(False)
        D[0][bidder_id][0] = np.append(D[0][bidder_id][0], bundle_to_add.reshape(1, -1), axis=0)
        D[0][bidder_id][1] = np.append(D[0][bidder_id][1], value_to_add)
        self.elicited_bids[economy_key] = D
        return(True)

    def check_bundle_contained(self, argmax_bundle, bidder_id):
        for economy_key, bidder_ids in self.economies_names.items():
            if bidder_id in bidder_ids:
                if any(np.equal(self.elicited_bids[economy_key][0][bidder_id][0], argmax_bundle).all(1)):
                    logging.debug('Argmax bundle ALREADY QUERIED IN DIFFERENT ECONOMY from {}'.format(bidder_id))
                    return(True)
        return(False)

    def update_weights(self, economy_key, bidder_id, weights):
        if len(weights) != self.elicited_bids[economy_key][0][bidder_id][1].shape[0]:
            raise ValueError('Wrong dimension')
        else:
            self.weights[economy_key][bidder_id] = weights

    def update_main_economy(self):
        for economy_key, bidder_ids in self.economies_names.items():
                if economy_key == 'Main Economy':
                    continue
                else:
                    logging.debug(economy_key)
                    for bidder_id in bidder_ids:
                        logging.debug(bidder_id)
                        bundle = self.argmax_allocation[economy_key][bidder_id][0]
                        logging.debug(bundle)
                        value = self.argmax_allocation[economy_key][bidder_id][1]
                        logging.debug(value)
                        added = self.update_bids(economy_key='Main Economy', bidder_id=bidder_id, bundle_to_add=bundle, value_to_add=value)
                        if self.sample_weights_on and added and self.status['Main Economy']:
                            pred_value = self.NN_models['Main Economy'][bidder_id].model.predict(bundle.reshape(1, -1), batch_size=self.batch_size).flatten()[0]
                            logging.debug('True: %s', value)
                            logging.debug('Predicted: %s', pred_value)
                            w = max(np.abs(value-pred_value)/self.weight_scaling['Main Economy'][bidder_id], 1)
                            logging.debug('Correction Weight: %s', w)
                            self.weights['Main Economy'][bidder_id] = np.concatenate([self.weights['Main Economy'][bidder_id], np.array([w])])
                    if self.sample_weights_on:
                        self.reset_weights(economy_key)

    def add_all_bids_to_main_economy(self):
        for economy_key, bidder_ids in self.economies_names.items():
            if economy_key == 'Main Economy':
                continue
            else:
                logging.debug(economy_key)
                for bidder_id in bidder_ids:
                    logging.debug(bidder_id)
                    for bids in range(0, self.bounds[economy_key][bidder_id]):
                        bundle = self.elicited_bids[economy_key][0][bidder_id][0][self.c0+bids+1, :]
                        logging.debug(bundle)
                        value = self.elicited_bids[economy_key][0][bidder_id][1][self.c0+bids+1]
                        logging.debug(value)
                        self.update_bids(economy_key='Main Economy', bidder_id=bidder_id, bundle_to_add=bundle, value_to_add=value)

    def _fit_NNs(self, Data, parameters, economy_key, sample_weights=None):
        Models = OrderedDict()
        D = Data[0]
        fitted_scaler = Data[1]
        for bidder_id, bids in D.items():
            logging.debug(bidder_id)
            # fit DNNs
            start = time.time()
            neural_net = NN(model_parameters=parameters[bidder_id], X_train=bids[0], Y_train=bids[1], scaler=fitted_scaler)
            neural_net.initialize_model(regularization_type=self.regularization_type)
            if sample_weights is not None:
                self.losses[economy_key][bidder_id].append(neural_net.fit(epochs=self.epochs, batch_size=self.batch_size, X_valid=None, Y_valid=None, sample_weight=sample_weights[bidder_id]))
            else:
                self.losses[economy_key][bidder_id].append(neural_net.fit(epochs=self.epochs, batch_size=self.batch_size, X_valid=None, Y_valid=None))
            Models[bidder_id] = neural_net
            end = time.time()
            logging.debug('Time for ' + bidder_id + ': %s sec\n', round(end-start))
        return(Models)

    def calculate_argmax_allocation(self, economy_key):
        # LINE 4: ESTIMATION STEP
        logging.debug('(1.1) ESTIMATON STEP')
        logging.debug('-----------------------------------------------')
        if self.sample_weights_on:
            self.NN_models[economy_key] = self._fit_NNs(Data=self.elicited_bids[economy_key], parameters=self.NN_parameters[economy_key], economy_key=economy_key, sample_weights=self.weights[economy_key])
            self.reset_weights(economy_key=economy_key)
        else:
            self.NN_models[economy_key] = self._fit_NNs(Data=self.elicited_bids[economy_key], parameters=self.NN_parameters[economy_key], economy_key=economy_key)
            self.reset_weights(economy_key=economy_key)
        DNNs = OrderedDict(list((key, self.NN_models[economy_key][key].model) for key in list(self.NN_models[economy_key].keys())))
        # LINE 5: OPTIMIZATION STEP
        logging.debug('(1.2) OPTIMIZATION STEP')
        logging.debug('-----------------------------------------------')
        for attempt in range(0, 5):
            logging.debug('Initialize MIP')
            X = NN_MIP(DNNs, L=self.L)
            if not self.Mip_bounds_tightening:
                X.initialize_mip(verbose=False)
            elif self.Mip_bounds_tightening == 'IA':
                X.tighten_bounds_IA(upper_bound_input=[1]*self.M)
                X.print_upper_bounds(only_zeros=True)
                X.initialize_mip(verbose=False)
            elif self.Mip_bounds_tightening == 'LP':
                X.tighten_bounds_IA(upper_bound_input=[1]*self.M)
                X.tighten_bounds_LP(upper_bound_input=[1]*self.M)
                X.print_upper_bounds(only_zeros=True)
                X.initialize_mip(verbose=False)
            try:
                logging.debug('Solving MIP')
                logging.debug('attempt no: %s', attempt)
                if self.warm_start and self.warm_start_sol[economy_key] is not None:
                    logging.debug('Using warm start')
                    self.warm_start_sol[economy_key] = X.solve_mip(log_output=False, time_limit=3600, mip_relative_gap=0.001, mip_start=docplex.mp.solution.SolveSolution(X.Mip, self.warm_start_sol[economy_key].as_dict()))
                    logging.debug(X.Mip.get_solve_status())
                else:
                    self.warm_start_sol[economy_key] = X.solve_mip(log_output=False, time_limit=3600, mip_relative_gap=0.001)
                    logging.debug(X.Mip.get_solve_status())
                b = 0
                for bidder_id in list(self.argmax_allocation[economy_key].keys()):
                    self.argmax_allocation[economy_key][bidder_id][0] = X.x_star[b, :]
                    b = b + 1
                logging.debug('ARGMAX ALLOCATION WAS SET')
                self.elapsed_time_mip[economy_key].append(X.soltime)
                break
            except Exception as e:
                logging.debug('Not solved succesfully')
                logging.debug(e)
                logging.debug(X)
                if attempt == 4:
                    X.Mip.export_as_lp(path=PATH,
                                       basename='UNSOLVED MIP in Iteration {} from economy'.format(self.iteration[economy_key]) + re.sub('[{}]', '', economy_key), hide_user_names=False)
                    sys.exit('STOP, not solved succesfully in {} attempts'.format(attempt+1))
                K.clear_session()
                # LINE 4: RE-FITTING STEP
                logging.debug('Refitting DNNs:')
                if self.sample_weights_on:
                    self.NN_models[economy_key] = self._fit_NNs(Data=self.elicited_bids[economy_key], parameters=self.NN_parameters[economy_key], economy_key=economy_key, sample_weights=self.weights[economy_key])
                else:
                    self.NN_models[economy_key] = self._fit_NNs(Data=self.elicited_bids[economy_key], parameters=self.NN_parameters[economy_key], economy_key=economy_key)
            DNNs = OrderedDict(list((key, self.NN_models[economy_key][key].model) for key in list(self.NN_models[economy_key].keys())))
        del X
        del DNNs

    def do_query(self, economy_key, allocation):  # WORKING PAPER VERSION WITH TOTAL CAPS PER PVM RUN AND NEW ALLOCATION RULE (+ "min_iteration", "reweighting" and "add marginals" optional)
        # LINE 6-13: QUERYING STEP
        STOP = 0
        logging.debug('\n(1.3) QUERYING STEP')
        logging.debug('-----------------------------------------------')
        for bidder_id in self.economies_names[economy_key]:
            argmax_bundle = allocation[economy_key][bidder_id][0]
            # If bundle is already asked from a bidder => next
            if any(np.equal(self.elicited_bids[economy_key][0][bidder_id][0], argmax_bundle).all(1)):
                logging.debug('Argmax bundle from iteration {} ALREADY ELICITATED from  {}'.format(self.iteration[economy_key], bidder_id))
            # If bound on allowed queries for an individual bidder is reached => next
            elif (self.total_bounds[bidder_id] >= self.ce):  # VERSION WITH TOTAL BOUNDS
                logging.debug('BOUND REACHED at iteration {} from {}'.format(self.iteration[economy_key], bidder_id))
            # If bound NOT reached and bundle NOT already queried => query value
            else:
                self.bounds[economy_key][bidder_id] = self.bounds[economy_key][bidder_id] + 1
                self.total_bounds[bidder_id] = self.total_bounds[bidder_id] + 1
                # increase actual queries only if bidder has never been asked this query in any economy
                if not self.check_bundle_contained(argmax_bundle, bidder_id):
                    logging.debug('Argmax bundle NEEDS TO BE QUERIED at iteration {} from {}'.format(self.iteration[economy_key], bidder_id))
                    self.actual_queries_elicitation[bidder_id] = self.actual_queries_elicitation[bidder_id] + 1
                logging.debug('Argmax bundle from iteration {} WILL BE ADDED TO ELICITATED BIDS from {}'.format(self.iteration[economy_key], bidder_id))

                value = self.value_model.calculate_value(int(list(bidder_id)[-1]), list(argmax_bundle.astype(int)))
                if self.fitted_scaler[economy_key] is not None:
                    logging.debug('Queried Value scaled by: %s', self.fitted_scaler[economy_key].scale_)
                    value = float(self.fitted_scaler[economy_key].transform([[value]]))
                self.argmax_allocation[economy_key][bidder_id][1] = value  # update argmax allocation value, bundle already set after optimization
                self.update_bids(economy_key=economy_key, bidder_id=bidder_id, bundle_to_add=self.argmax_allocation[economy_key][bidder_id][0], value_to_add=value)  # update bids
                # update weights
                if self.sample_weights_on:
                    pred_value = self.NN_models[economy_key][bidder_id].model.predict(argmax_bundle.reshape(1, -1), batch_size=self.batch_size).flatten()[0]
                    logging.debug('True: %s', value)
                    logging.debug('Predicted: %s', pred_value)
                    w = max(np.abs(value-pred_value)/self.weight_scaling[economy_key][bidder_id], 1)
                    logging.debug('Correction Weight: %s', w)
                    self.update_weights(economy_key=economy_key, bidder_id=bidder_id, weights=np.concatenate([np.ones(self.elicited_bids[economy_key][0][bidder_id][0].shape[0]-1), np.array([w])]))
                STOP = STOP + 1
        # stopping criteria if all bundles of the argmax allocation were already elicited AND at least min_iteration iterations were conducted
        if STOP == 0 and self.iteration[economy_key] >= self.min_iteration:
            logging.debug('\nArgmax bundles in economy {} from iteration {} already elicited from ALL bidders => STOP'.format(economy_key, self.iteration[economy_key]))
            self.status[economy_key] = False
            return(None)
        # stopping criteria if all bundles of the argmax allocation were already elicited BUT less than min_iteration iterations were conducted, thus query random bundles
        if STOP == 0 and self.iteration[economy_key] < self.min_iteration and any([total_bound < self.ce for bidder, total_bound in self.total_bounds.items()]):  # VERSION WITH TOTAL BOUNDS
            logging.debug('\nArgmax bundles in economy {} from iteration {} already elicited from ALL bidders BUT {} < {} => CONTINUE'.format(economy_key, self.iteration[economy_key],
                          self.iteration[economy_key], self.min_iteration))
            # query random bundles from EACH bidder, adjust the weights and save the random bundle-value pairs to the argmax_allocation
            for bidder_id in self.economies_names[economy_key]:
                if self.total_bounds[bidder_id] < self.ce:  # VERSION WITH TOTAL BOUNDS
                    self.bounds[economy_key][bidder_id] = self.bounds[economy_key][bidder_id] + 1
                    self.total_bounds[bidder_id] = self.total_bounds[bidder_id] + 1
                    # tmp = np.array(self.value_model.get_random_bids(bidder_id=int(list(bidder_id)[-1]), number_of_bids=1)[0])  #SATS uniform
                    tmp = np.random.choice(2, size=self.M)   # TRUE uniform 1.
                    tmp = np.append(tmp, self.value_model.calculate_value(bidder_id=bidder_id, goods_vector=tmp))  # TRUE uniform 2.
                    random_bundle = tmp[:-1]
                    value = tmp[-1]
                    # increase actual queries only if bidder has never been asked this query in any economy
                    if not self.check_bundle_contained(random_bundle, bidder_id):
                        logging.debug('Random bundle NEEDS TO BE QUERIED at iteration {} from {}'.format(self.iteration[economy_key], bidder_id))
                        self.actual_queries_elicitation[bidder_id] = self.actual_queries_elicitation[bidder_id] + 1
                    logging.debug('Random bundle from iteration {} WILL BE ADDED TO ELICITATED BIDS from {}'.format(self.iteration[economy_key], bidder_id))
                    if self.fitted_scaler[economy_key] is not None:
                        logging.debug('Queried Value scaled by: %s', self.fitted_scaler[economy_key].scale_)
                        value = float(self.fitted_scaler[economy_key].transform([[value]]))
                    self.argmax_allocation[economy_key][bidder_id][0] = random_bundle  # overwrite argmax bundle with random bundle
                    self.argmax_allocation[economy_key][bidder_id][1] = value  # overwrite None with value of random bundle
                    self.update_bids(economy_key=economy_key, bidder_id=bidder_id, bundle_to_add=random_bundle, value_to_add=value)  # update bids
                    # update weights
                    if self.sample_weight_on:
                        pred_value = self.NN_models[economy_key][bidder_id].model.predict(random_bundle.reshape(1, -1), batch_size=self.batch_size).flatten()[0]
                        logging.debug('True: %s', value)
                        logging.debug('Predicted: %s', pred_value)
                        w = max(np.abs(value-pred_value)/self.weight_scaling[economy_key][bidder_id], 1)
                        logging.debug('Correction Weight: %s', w)
                        self.update_weights(economy_key=economy_key, bidder_id=bidder_id, weights=np.concatenate([np.ones(self.elicited_bids[economy_key][0][bidder_id][0].shape[0]-1), np.array([w])]))
        # VERSION WITH TOTAL BOUNDS
        # check if all bounds were reached and in this case terminate all economies
        if all([total_bound >= self.ce for bidder, total_bound in self.total_bounds.items()]):
            logging.debug('Total Bounds across Economies: ' + '  '.join("{}:{}".format(k, v) for k, v in self.total_bounds.items()) + '\n')
            logging.debug('ALL Total Bounds reached in economy {} from iteration {} => STOP'.format(economy_key, self.iteration[economy_key]))
            for economy_key in list(self.economies_names.keys()):
                self.status[economy_key] = False
            return(None)
        # update iterations of economy
        self.iteration[economy_key] = self.iteration[economy_key] + 1
        return(None)

    def do_query_2(self, economy_key, allocation):  # IJCAI VERSION WITH INDIVIDUAL CAPS PER ELICITATION RUN (changes to version do_query marked with ***)
        # LINE 6-13: QUERYING STEP
        STOP = 0
        logging.debug('\n(1.3) QUERYING STEP')
        logging.debug('-----------------------------------------------')
        for bidder_id in self.economies_names[economy_key]:
            argmax_bundle = allocation[economy_key][bidder_id][0]
            # If bundle is already asked from a bidder => next
            if any(np.equal(self.elicited_bids[economy_key][0][bidder_id][0], argmax_bundle).all(1)):
                logging.debug('Argmax bundle from iteration {} ALREADY ELICITATED from  {}'.format(self.iteration[economy_key], bidder_id))
            # If bound on allowed queries for an individual bidder is reached => next
            elif (self.bounds[economy_key][bidder_id] >= self.ce):  # VERSION WITH INDIVIDUAL BOUNDS ***
                logging.debug('BOUND REACHED at iteration {} from {}'.format(self.iteration[economy_key], bidder_id))
            # If bound NOT reached and bundle NOT already queried => query value
            else:
                self.bounds[economy_key][bidder_id] = self.bounds[economy_key][bidder_id] + 1
                self.total_bounds[bidder_id] = self.total_bounds[bidder_id] + 1
                # increase actual queries only if bidder has never been asked this query in any economy
                if not self.check_bundle_contained(argmax_bundle, bidder_id):
                    logging.debug('Argmax bundle NEEDS TO BE QUERIED at iteration {} from {}'.format(self.iteration[economy_key], bidder_id))
                    self.actual_queries_elicitation[bidder_id] = self.actual_queries_elicitation[bidder_id] + 1
                logging.debug('Argmax bundle from iteration {} WILL BE ADDED TO ELICITATED BIDS from {}'.format(self.iteration[economy_key], bidder_id))

                value = self.value_model.calculate_value(int(list(bidder_id)[-1]), list(argmax_bundle.astype(int)))
                if self.fitted_scaler[economy_key] is not None:
                    logging.debug('Queried Value scaled by: %s', self.fitted_scaler[economy_key].scale_)
                    value = float(self.fitted_scaler[economy_key].transform([[value]]))
                self.argmax_allocation[economy_key][bidder_id][1] = value  # update argmax allocation value, bundle already set after optimization
                self.update_bids(economy_key=economy_key, bidder_id=bidder_id, bundle_to_add=self.argmax_allocation[economy_key][bidder_id][0], value_to_add=value)  # update bids
                # update weights
                if self.sample_weights_on:
                    pred_value = self.NN_models[economy_key][bidder_id].model.predict(argmax_bundle.reshape(1, -1), batch_size=self.batch_size).flatten()[0]
                    logging.debug('True: %s', value)
                    logging.debug('Predicted: %s', pred_value)
                    w = max(np.abs(value-pred_value)/self.weight_scaling[economy_key][bidder_id], 1)
                    logging.debug('Correction Weight: %s', w)
                    self.update_weights(economy_key=economy_key, bidder_id=bidder_id, weights=np.concatenate([np.ones(self.elicited_bids[economy_key][0][bidder_id][0].shape[0]-1), np.array([w])]))
                STOP = STOP + 1
        # stopping criteria if all bundles of the argmax allocation were already elicited AND at least min_iteration iterations were conducted
        if STOP == 0 and self.iteration[economy_key] >= self.min_iteration:
            logging.debug('\nArgmax bundles in economy {} from iteration {} already elicited from ALL bidders => STOP'.format(economy_key, self.iteration[economy_key]))
            self.status[economy_key] = False
            return(None)
        # stopping criteria if all bundles of the argmax allocation were already elicited BUT less than min_iteration iterations were conducted, thus query random bundles
        if STOP == 0 and self.iteration[economy_key] < self.min_iteration and any([bound < self.ce for bidder, bound in self.bounds[economy_key].items()]):  # VERSION WITH INDIVIDUAL BOUNDS ***
            logging.debug('\nArgmax bundles in economy {} from iteration {} already elicited from ALL bidders BUT {} < {} => CONTINUE'.format(economy_key, self.iteration[economy_key],
                          self.iteration[economy_key], self.min_iteration))
            # query random bundles from EACH bidder, adjust the weights and save the random bundle-value pairs to the argmax_allocation
            for bidder_id in self.economies_names[economy_key]:
                if self.bounds[economy_key][bidder_id] < self.ce:  # VERSION WITH INDIVIDUAL BOUNDS ***
                    self.bounds[economy_key][bidder_id] = self.bounds[economy_key][bidder_id] + 1
                    self.total_bounds[bidder_id] = self.total_bounds[bidder_id] + 1
                    # tmp = np.array(self.value_model.get_random_bids(bidder_id=int(list(bidder_id)[-1]), number_of_bids=1)[0]) # SATS uniform
                    tmp = np.random.choice(2, size=self.M)   # TRUE uniform 1.
                    tmp = np.append(tmp, self.value_model.calculate_value(bidder_id=bidder_id, goods_vector=tmp))  # TRUE uniform 2.
                    random_bundle = tmp[:-1]
                    value = random_bundle[-1]
                    # increase actual queries only if bidder has never been asked this query in any economy
                    if not self.check_bundle_contained(random_bundle, bidder_id):
                        logging.debug('Random bundle NEEDS TO BE QUERIED at iteration {} from {}'.format(self.iteration[economy_key], bidder_id))
                        self.actual_queries_elicitation[bidder_id] = self.actual_queries_elicitation[bidder_id] + 1
                    logging.debug('Random bundle from iteration {} WILL BE ADDED TO ELICITATED BIDS from {}'.format(self.iteration[economy_key], bidder_id))
                    if self.fitted_scaler[economy_key] is not None:
                        logging.debug('Queried Value scaled by: %s', self.fitted_scaler[economy_key].scale_)
                        value = float(self.fitted_scaler[economy_key].transform([[value]]))
                    self.argmax_allocation[economy_key][bidder_id][0] = random_bundle  # overwrite argmax bundle with random bundle
                    self.argmax_allocation[economy_key][bidder_id][1] = value  # overwrite None with value of random bundle
                    self.update_bids(economy_key=economy_key, bidder_id=bidder_id, bundle_to_add=random_bundle, value_to_add=value)  # update bids
                    # update weights
                    if self.sample_weight_on:
                        pred_value = self.NN_models[economy_key][bidder_id].model.predict(random_bundle.reshape(1, -1), batch_size=self.batch_size).flatten()[0]
                        logging.debug('True: %s', value)
                        logging.debug('Predicted: %s', pred_value)
                        w = max(np.abs(value-pred_value)/self.weight_scaling[economy_key][bidder_id], 1)
                        logging.debug('Correction Weight: %s', w)
                        self.update_weights(economy_key=economy_key, bidder_id=bidder_id, weights=np.concatenate([np.ones(self.elicited_bids[economy_key][0][bidder_id][0].shape[0]-1), np.array([w])]))
        # VERSION WITH INDIVIDUAL BOUNDS ***
        # check if bounds in specific economy was reached and in this case terminate all economies
        if all([bound >= self.ce for bidder, bound in self.bounds[economy_key].items()]):
            logging.debug('Individual Bounds in economy {}: ' + '  '.join("{}:{}".format(economy_key, k, v) for k, v in self.bounds[economy_key].items()) + '\n')
            logging.debug('ALL Individual Bounds in economy {} reached in iteration {} => STOP'.format(economy_key, self.iteration[economy_key]))
            self.status[economy_key] = False
            return(None)
        # update iterations of economy
        self.iteration[economy_key] = self.iteration[economy_key] + 1
        return(None)

    def solve_WDP(self, economy_key=None):
        if economy_key is None:
            for economy_key in list(self.economies_names.keys()):
                logging.debug('Solving WDP based on elicited Bids for economy {}:'.format(economy_key))
                elicited_bids = [np.concatenate((bids[0], bids[1].reshape(-1, 1)), axis=1) for bidder_id, bids in self.elicited_bids[economy_key][0].items()]
                start = time.time()
                X = WDP(elicited_bids)
                X.initialize_mip(verbose=False)
                X.solve_mip()
                allocation_PA = util_github.format_solution_MIP_new(Mip=X.Mip, elicited_bids=elicited_bids, bidder_names=self.economies_names[economy_key], fitted_scaler=self.fitted_scaler[economy_key])
                value_PA = X.Mip.objective_value
                if self.fitted_scaler[economy_key] is not None:
                    logging.debug('Queried Value scaled by: %s', self.fitted_scaler[economy_key].scale_)
                    value_PA = float(self.fitted_scaler[economy_key].inverse_transform([[value_PA]]))
                end = time.time()
                self.WDP_allocations[economy_key] = [allocation_PA, value_PA]
                logging.debug('WDP with elicited Bids solved in ' + time.strftime('%H:%M:%S', time.gmtime(end-start)) + '\n')
                logging.debug('Allocation:')
                for key in list(allocation_PA.keys()):
                    logging.debug('%s %s', key, allocation_PA[key])
                logging.debug('Value: %s \n', value_PA)
                del allocation_PA
                del value_PA
                del elicited_bids
                del X
        else:
            logging.debug('Solving WDP based on elicited Bids for economy {}:'.format(economy_key))
            elicited_bids = [np.concatenate((bids[0], bids[1].reshape(-1, 1)), axis=1) for bidder_id, bids in self.elicited_bids[economy_key][0].items()]
            start = time.time()
            X = WDP(elicited_bids)
            X.initialize_mip(verbose=False)
            X.solve_mip()
            allocation_PA = util_github.format_solution_MIP_new(Mip=X.Mip, elicited_bids=elicited_bids, bidder_names=self.economies_names[economy_key], fitted_scaler=self.fitted_scaler[economy_key])
            value_PA = X.Mip.objective_value
            if self.scaler is not None:
                logging.debug('Mip Objective Value scaled by: %s', self.fitted_scaler[economy_key].scale_)
                value_PA = float(self.fitted_scaler[economy_key].inverse_transform([[value_PA]]))
            end = time.time()
            self.WDP_allocations[economy_key] = [allocation_PA, value_PA]
            logging.debug('WDP with elicited Bids solved in ' + time.strftime('%H:%M:%S', time.gmtime(end-start)) + '\n')
            logging.debug('Allocation:')
            for key in list(allocation_PA.keys()):
                logging.debug(allocation_PA[key])
            logging.debug('\nValue: %s \n', value_PA)
            del allocation_PA
            del value_PA
            del elicited_bids
            del X

    def calculate_finalpvm_alloc(self):
        max_economy = max(self.WDP_allocations.keys(), key=(lambda k: self.WDP_allocations[k][1]))
        logging.debug('PVM allocation obtained in economy: {}'.format(max_economy))
        self.pvm_allocation = self.WDP_allocations[max_economy]
        if max_economy != 'Main Economy':
            self.pvm_allocation[0]['Bidder_' + re.findall(r'\d+', max_economy)[0]] = {'good_ids': [], 'value': 0}  # add missing bidder if a marginal economy is the pvm allocation
        logging.debug('Final PVM Allocation:')
        for key in list(self.pvm_allocation[0].keys()):
            logging.debug('%s %s', key, self.pvm_allocation[0][key])
        logging.debug('\nValue: %s \n', self.pvm_allocation[1])

    def calculate_payments(self):  # here p_i^pvm:=p_1-p2
        for economy_key in list(self.economies_names.keys()):
            if economy_key == 'Main Economy':
                continue
            else:
                bidder = list(set(self.economies_names['Main Economy'])-set(self.economies_names[economy_key]))[0]  # determine which bidder is excluded in this economy
                p1 = self.WDP_allocations[economy_key][1]  # social value of the allocation x^(-i) in this economy
                # social value of main economy = pvm allocation (with new allocation rule) without bidder i DEPRECATED
                # p2 = sum([self.WDP_allocations['Main Economy'][0][bidder_id]['value'] for bidder_id in self.economies_names[economy_key]])
                p2 = sum([self.pvm_allocation[0][bidder_id]['value'] for bidder_id in self.economies_names[economy_key]])  # social value of pvm allocation without bidder i NEW
                self.payments[bidder] = round(p1-p2, 2)
                logging.debug('Payment %s: %s - %s  =  %s \n', bidder, p1, p2, self.payments[bidder])
