# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:40:12 2020

@author: jakob
"""

"""
FILE DESCRIPTION:

This file implements the class Economies. This class is used for simulating the DNN-based ICA, i.e., the PVM mechanism (Algorithm 2 in Section 2), based on DNNs (Deep Neural Networks).
Specifically, this class stores all the information about the economies in the iterative procedure of the DNN-based preference elicitation algorithm (Algorithm 1 in Section 2).

Economies has the following functionalities:
    0.CONSTRUCTOR: __init__(self,  SATS_auction_instance, SATS_auction_instance_seed, Qinit, Qmax, Qround, scaler)
        SATS_auction_instance = auction instance created with SATS (GSVM, LSVM, MRVM)
        SATS_auction_instance_seed = seed of SATS auction instance
        Qinit = number of intial data points, i.e., bundle-value pairs (queried randomly per bidder prior to the elicitation procedure) (positive integer)
        Qmax = maximal number of possible value queries per bidder (positive integer)
        Qround = number of sampled marginals per bidder per auction round == number of value queries per bidder per auction round
        scaler = scaler is a instance from sklearn.MinMaxScaler() used for scaling the values of the bundle-value pairs
        This method creates a instance of the class Economies. Details about the attributes can be found below.
    1.METHOD: get_info(self, final_summary=False)
        This method retunrs information of the current status of the MLCA run.
    2.METHOD: get_number_of_elicited_bids(self, bidder=None)
        This method gets the number of elicited bids so far
    3.METHOD: calculate_efficiency_per_iteration(self):
        Calculates efficient allocation per acution round and corresponding efficiency
    4.METHOD: set_NN_parameters(self, parameters)
    5.METHOD: set_MIP_parameters(self, parameters)
    6.METHOD: set_initial_bids(self, initial_bids=None, fitted_scaler=None)
        initial_bids = self defined initial bids
        fitted_scaler = corresponding fitted scaler instance
        This method creates sets of initial bundle-value pairs, sets the elicited bids attribute and the scaler attribute, i.e., bids.
        If no self defnied initial_bids are given it calls the function util.initial_bids_mlca_unif (uniformly at random from the bundle space 2^m).
    4.METHOD: reset_argmax_allocations(self)
        This method cleares the argmax allocation a^(t) from the round t.
    5.METHOD: reset_current_query_profile(self)
        This method resets the current query profile S
    6.METHOD: reset_NN_models(self)
        This method resets the NN keras models, i.e., the DNNs from the previous round.
    7.METHOD: reset_economy_status(self)
        This method resets the economy staus, if an unrestricted allocation has already been calculated for this economy
    8.METHOD: solve_SATS_auction_instance(self)
        This method solves thegiven SATS instance for the true efficient allocation.
    9.METHOD: sample_marginal_economies(self, active_bidder, number_of_marginals)
        This method samples randomly marginal economies for the active_bidder
    10.METHOD: update_elicited_bids(self)
        This method adds the bids from the current query profile to the elicited bids
    11.METHOD: update_current_query_profile(self, bidder, bundle_to_add)
        This method  updates the current query profile for a bidder with a bundle_to_add
    12.METHOD: value_queries(self, bidder_id, bundles)
        This method performs a value query for a bidder with bidder_id and several bundles
    13.METHOD: check_bundle_contained(self, bundle, bidder)
        This method checks if the bundle already has been queried from bidder in the elicited bids or the current query profile.
    14.METHOD: next_queries(self, economy_key, active_bidder)
        This method performs the nextQueries algorithm for the bidder active_bidder and the economy economy_key
    15.METHOD: estimation_step(self, economy_key)
        This method performs the estimation step for a by economy_key specified economy
    16.METHOD: optimization_step(self, economy_key, bidder_specific_constraints=None)
        This method performs the optimization step for a by a economy_key specified economy.
        If bidder_specific_constraints is not None this solves the additional bidder specific MIP with constraints given by the elicited bids and the current query profile of the specific bidder.
        bidder_specific_constraints is a dict with (bidder name, np.array of specific bundles that serve as constraints)
    17.METHOD: calculate_mlca_allocation(self, economy='Main Economy')
        This method calculates the MLCA allocation given elicited bids for any economy speified.
    18.METHOD: solve_WDP(self, elicited_bids, verbose=0)
        This method solves the WDP  given elicited_bids.
    19.METHOD: calculate_efficiency_of_allocation(self, allocation, allocation_scw, verbose=0)
        This method calculates the efficiency w.r.t to the true efficient SATS allocation
    19.METHOD: calculate_vcg_payments(self, forced_recalc=False)
        This method calculates the VCG-style payments given current elicited bids.

See example_mlca_economies.py for an example of how to use the class Economies.
"""

# Libs
import os
import itertools
import sys
import numpy as np
import random
import time
from tensorflow.keras.backend import clear_session
from collections import OrderedDict
import logging
# CPLEX: Here, DOcplex is used for solving the deep neural network-based Winner Determination Problem.
import docplex
# documentation
# http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html

# Own Modules
import source.mlca.mlca_util as util
from source.mlca.mlca_nn import MLCA_NN
from source.mlca.mlca_nn_mip import MLCA_NNMIP
from source.mlca.mlca_wdp import MLCA_WDP


__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'

# %%


class MLCA_Economies:

    def __init__(self, SATS_auction_instance, SATS_auction_instance_seed, Qinit, Qmax, Qround, scaler):

        # STATIC ATTRIBUTES
        self.SATS_auction_instance = SATS_auction_instance  # auction instance from SATS: LSVM, GSVM or MRVM generated via PySats.py.
        self.SATS_auction_instance_allocation = None  # true efficient allocation of auction instance
        self.SATS_auction_instance_scw = None  # social welfare of true efficient allocation of auction instance
        self.SATS_auction_instance_seed = SATS_auction_instance_seed # auction instance seed from SATS
        self.bidder_ids = list(SATS_auction_instance.get_bidder_ids())  # bidder ids in this auction instance.
        self.bidder_names = list('Bidder_{}'.format(bidder_id) for bidder_id in self.bidder_ids)
        self.N = len(self.bidder_ids)  # number of bidders
        self.good_ids = set(SATS_auction_instance.get_good_ids())  # good ids in this auction instance
        self.M = len(self.good_ids)  # number of items
        self.Qinit = Qinit  # number of intial data points, i.e., bundle-value pairs (queried randomly per bidder prior to the elicitation procedure, different per bidder)
        self.Qmax = Qmax  # maximal number of possible value queries in the preference elicitation algorithm (PEA) per bidder
        self.Qround = Qround  # maximal number of marginal economies per auction round per bidder
        self.scaler = scaler  # scaler is a instance from sklearn.MinMaxScaler() used for scaling the values of the bundle-value pairs
        self.fitted_scaler = None  # fitted scaler to the initial bids
        self.mlca_allocation = None  # mlca allocation
        self.mlca_scw = None # true social welfare of mlca allocation
        self.mlca_allocation_efficiency = None # efficiency of mlca allocation
        self.MIP_parameters = None  # MIP parameters
        self.mlca_iteration = 0 # mlca iteration tracker
        subsets = list(map(list, itertools.combinations(self.bidder_ids, self.N-1))) # orderedDict containing all economies and the corresponding bidder ids of the bidders which are active in these economies.
        subsets.sort(reverse=True) #sort s.t. Marginal Economy (-0)-> Marginal Economy (-1) ->...
        self.economies = OrderedDict(list(('Marginal Economy -({})'.format(i), econ) for econ, i in zip(subsets, [[x for x in self.bidder_ids if x not in subset][0] for subset in subsets])))
        self.economies['Main Economy'] = self.bidder_ids
        self.economies_names = OrderedDict(list((key, ['Bidder_{}'.format(s) for s in value]) for key, value in self.economies.items())) # orderedDict containing all economies and the corresponding bidder names (as strings) of the bidders which are active in these economies.
        self.efficiency_per_iteration = OrderedDict()  #storage for efficiency stat püer auction round
        self.efficient_allocation_per_iteration = OrderedDict()  # storage for efficent allocation per auction round

        # DYNAMIC PER ECONOMY
        self.economy_status = OrderedDict(list((key, False) for key, value in self.economies.items()))  # boolean, status of economy: if already calculated.
        self.mlca_marginal_allocations = OrderedDict(list((key, None) for key, value in self.economies.items() if key!='Main Economy'))  # Allocation of the WDP based on the elicited bids
        self.mlca_marginal_scws = OrderedDict(list((key, None) for key, value in self.economies.items() if key!='Main Economy'))  # Social Welfare of the Allocation of the WDP based on the elicited bids
        self.elapsed_time_mip = OrderedDict(list((key, []) for key, value in self.economies.items()))  # stored MIP solving times per economy
        self.warm_start_sol = OrderedDict(list((key, None) for key, value in self.economies.items()))  # MIP SolveSolution object used as warm start per economy

        # DYNAMIC PER BIDDER
        self.mlca_payments = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))  # VCG-style payments in MLCA, calculated at the end
        self.elicited_bids = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))  # R=(R_1,...,R_n) elicited bids per bidder
        self.current_query_profile = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids)) # S=(S_1,...,S_n) number of actual value queries, that takes into account if the same bundle was queried from a bidder in two different economies that it is not counted twice
        self.NN_parameters = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))  # DNNs parameters as in the Class NN described.

        # DYNAMIC PER ECONOMY & BIDDER
        self.argmax_allocation = OrderedDict(list((key, OrderedDict(list((bidder_id, [None, None]) for bidder_id in value))) for key, value in self.economies_names.items()))  # [a,a_restr] a: argmax bundles per bidder and economy, a_restr: restricted argmax bundles per bidder and economy
        self.NN_models = OrderedDict(list((key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in self.economies_names.items()))  # tf.keras NN models
        self.losses = OrderedDict(list((key, OrderedDict(list((bidder_id, []) for bidder_id in value))) for key, value in self.economies_names.items()))  # Storage for the MAE loss during training of the DNNs

    def get_info(self, final_summary=False):
        if not final_summary: logging.warning('INFO')
        if final_summary: logging.warning('SUMMARY')
        logging.warning('-----------------------------------------------')
        logging.warning('Seed Auction Instance: %s', self.SATS_auction_instance_seed)
        logging.warning('Iteration of MLCA: %s', self.mlca_iteration)
        logging.warning('Number of Elicited Bids:')
        for k,v in self.elicited_bids.items():
            logging.warning(k+': %s', v[0].shape[0]-1)  # -1 because of added zero bundle
        logging.warning('Qinit: %s | Qround: %s | Qmax: %s', self.Qinit, self.Qround, self.Qmax)
        if not final_summary: logging.warning('Efficiency given elicited bids from iteration 0-%s: %s\n', self.mlca_iteration-1,self.efficiency_per_iteration[self.mlca_iteration-1])

    def get_number_of_elicited_bids(self, bidder=None):
        if bidder is None:
            return OrderedDict((bidder,self.elicited_bids[bidder][0].shape[0]-1) for bidder in self.bidder_names)  # -1, because null bundle was added
        else:
            return self.elicited_bids[bidder][0].shape[0]-1

    def calculate_efficiency_per_iteration(self):
        logging.debug('')
        logging.debug('Calculate current efficiency:')
        allocation, objective = self.solve_WDP(self.elicited_bids, verbose=1)
        self.efficient_allocation_per_iteration[self.mlca_iteration] = allocation
        efficiency = self.calculate_efficiency_of_allocation(allocation=allocation, allocation_scw=objective) # -1 elicited bids after previous iteration
        self.efficiency_per_iteration[self.mlca_iteration] = efficiency

    def set_NN_parameters(self, parameters):
        logging.debug('Set NN parameters')
        self.NN_parameters = OrderedDict(parameters)

    def set_MIP_parameters(self, parameters):
        logging.debug('Set MIP parameters')
        self.MIP_parameters = parameters

    def set_initial_bids(self, initial_bids=None, fitted_scaler=None):
        logging.info('INITIALIZE BIDS')
        logging.info('-----------------------------------------------\n')
        if initial_bids is None: # Uniform sampling
            self.elicited_bids, self.fitted_scaler = util.initial_bids_mlca_unif(SATS_auction_instance=self.SATS_auction_instance,
                                                                                 number_initial_bids=self.Qinit, bidder_names=self.bidder_names,
                                                                                 scaler=self.scaler)
        else:
            logging.debug('Setting inputed initial bids of dimensions:')
            if not (list(initial_bids.keys()) == self.bidder_names):
                logging.info('Cannot set inputed initial bids-> sample uniformly.') # Uniform sampling
                self.elicited_bids, self.fitted_scaler = util.initial_bids_mlca_unif(SATS_auction_instance=self.SATS_auction_instance,
                                                                                 number_initial_bids=self.Qinit,bidder_names=self.bidder_names,
                                                                                 scaler=self.scaler)
            else:
                for k,v in initial_bids.items():
                    logging.debug(k + ': X=%s, Y=%s',v[0].shape, v[1].shape)
                self.elicited_bids = initial_bids # needed format: {Bidder_i:[bundles,values]}
                self.fitted_scaler = fitted_scaler # fitted scaler to initial bids
                self.Qinit = [v[0].shape[0] for k,v in initial_bids.items()]

    def reset_argmax_allocations(self):
        self.argmax_allocation = OrderedDict(list((key, OrderedDict(list((bidder_id, [None, None]) for bidder_id in value))) for key, value in self.economies_names.items()))

    def reset_current_query_profile(self):
        self.current_query_profile =  OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))

    def reset_NN_models(self):
        delattr(self, 'NN_models')
        self.NN_models = OrderedDict(list((key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in self.economies_names.items()))
        clear_session()

    def reset_economy_status(self):
        self.economy_status = OrderedDict(list((key, False) for key, value in self.economies.items()))

    def solve_SATS_auction_instance(self):
        self.SATS_auction_instance_allocation, self.SATS_auction_instance_scw = self.SATS_auction_instance.get_efficient_allocation()

    def sample_marginal_economies(self, active_bidder, number_of_marginals):
        admissible_marginals = [x for x in list((self.economies.keys())) if x not in ['Main Economy', 'Marginal Economy -({})'.format(active_bidder)]]
        return random.sample(admissible_marginals, k=number_of_marginals)

    def update_elicited_bids(self):
        for bidder in self.bidder_names:
            logging.info('UPDATE ELICITED BIDS: S -> R for %s', bidder)
            logging.info('---------------------------------------------')
            # update bundles
            self.elicited_bids[bidder][0] = np.append(self.elicited_bids[bidder][0],self.current_query_profile[bidder], axis=0)
            # update values
            bidder_value_reports = self.value_queries(bidder_id=util.key_to_int(bidder), bundles=self.current_query_profile[bidder])
            self.elicited_bids[bidder][1] = np.append(self.elicited_bids[bidder][1],bidder_value_reports, axis=0) # update values
            logging.info('CHECK Uniqueness of updated elicited bids:')
            check = len(np.unique(self.elicited_bids[bidder][0], axis=0))==len(self.elicited_bids[bidder][0])
            logging.info('UNIQUE\n') if check else logging.debug('NOT UNIQUE\n')
        return(check)

    def update_current_query_profile(self, bidder, bundle_to_add):
        if bundle_to_add.shape != (self.M,):
            logging.debug('No valid bundle dim -> CANNOT ADD BUNDLE')
            return(False)
        if self.current_query_profile[bidder] is None: # If empty can add bundle_to_add for sure
            logging.debug('Current query profile is empty -> ADD BUNDLE')
            self.current_query_profile[bidder] = bundle_to_add.reshape(1,-1)
            return(True)
        else: # If not empty, first check for duplicates, then add
            if self.check_bundle_contained(bundle=bundle_to_add, bidder=bidder):
                return(False)
            else:
                self.current_query_profile[bidder] = np.append(self.current_query_profile[bidder], bundle_to_add.reshape(1, -1), axis=0)
                logging.debug('ADD BUNDLE to current query profile')
                return(True)

    def value_queries(self, bidder_id, bundles):
        raw_values = np.array([self.SATS_auction_instance.calculate_value(bidder_id, bundles[k,:]) for k in range(bundles.shape[0])])
        if self.fitted_scaler is None:
            logging.debug('Return raw value queries')
            return (raw_values)
        else:
            minI = int(round(self.fitted_scaler.data_min_[0]*self.fitted_scaler.scale_[0]))
            maxI = int(round(self.fitted_scaler.data_max_[0]*self.fitted_scaler.scale_[0]))
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug('raw values %s', raw_values)
            logging.debug('Return value queries scaled by: %s to the interval [%s,%s]',round(self.fitted_scaler.scale_[0],8),minI,maxI)
            logging.debug('scaled values %s', self.fitted_scaler.transform(raw_values.reshape(-1,1)).flatten())
            logging.debug('---------------------------------------------')
            return (self.fitted_scaler.transform(raw_values.reshape(-1,1)).flatten())

    def check_bundle_contained(self, bundle, bidder):
        if np.any(np.equal(self.elicited_bids[bidder][0],bundle).all(axis=1)):
            logging.info('Argmax bundle ALREADY ELICITED from {}\n'.format(bidder))
            return(True)
        if self.current_query_profile[bidder] is not None:
            if np.any(np.equal(self.current_query_profile[bidder], bundle).all(axis=1)):
                logging.info('Argmax bundle ALREADY QUERIED IN THIS AUCTION ROUND from {}\n'.format(bidder))
                return(True)
        return(False)

    def next_queries(self, economy_key, active_bidder):
        if not self.economy_status[economy_key]:  # check if economy already has been calculated prior
            self.estimation_step(economy_key=economy_key)
            self.optimization_step(economy_key=economy_key)

        if self.check_bundle_contained(bundle=self.argmax_allocation[economy_key][active_bidder][0], bidder=active_bidder): # Check if argmax bundle already has been queried in R or S
            if self.current_query_profile[active_bidder] is not None: # recalc optimization step with bidder specific constraints added
                Ri_union_Si = np.append(self.elicited_bids[active_bidder][0], self.current_query_profile[active_bidder],axis=0)
            else:
                Ri_union_Si = self.elicited_bids[active_bidder][0]
            CTs = OrderedDict()
            CTs[active_bidder] = Ri_union_Si
            self.optimization_step(economy_key, bidder_specific_constraints=CTs)
            self.economy_status[economy_key] = True  # set status of economy to true
            return(self.argmax_allocation[economy_key][active_bidder][1])  # return constrained argmax bundle

        else: # If argmax bundle has NOT already been queried
            self.economy_status[economy_key]=True  # set status of economy to true
            return(self.argmax_allocation[economy_key][active_bidder][0]) # return regular argmax bundle

    def estimation_step(self, economy_key):
        logging.info('ESTIMATON STEP')
        logging.info('-----------------------------------------------')
        models = OrderedDict()
        for bidder in self.economies_names[economy_key]:
            bids = self.elicited_bids[bidder]
            logging.info(bidder)

            start = time.time()
            nn_model = MLCA_NN(X_train=bids[0], Y_train=bids[1], scaler=self.fitted_scaler)  # instantiate class
            nn_model.initialize_model(model_parameters=self.NN_parameters[bidder])  # initialize model
            tmp = nn_model.fit(epochs=self.NN_parameters[bidder]['epochs'], batch_size=self.NN_parameters[bidder]['batch_size'],  # fit model to data
                               X_valid=None, Y_valid=None)
            end = time.time()
            logging.info('Time for ' + bidder + ': %s sec\n', round(end-start))
            self.losses[economy_key][bidder].append(tmp)
            models[bidder] = nn_model
        self.NN_models[economy_key] = models

    def optimization_step(self, economy_key, bidder_specific_constraints=None):
        DNNs = OrderedDict(list((key, self.NN_models[economy_key][key].model) for key in list(self.NN_models[economy_key].keys())))

        if bidder_specific_constraints is None:
            logging.info('OPTIMIZATION STEP')
        else:
            logging.info('ADDITIONAL BIDDER SPECIFIC OPTIMIZATION STEP for {}'.format(list(bidder_specific_constraints.keys())[0]))
        logging.info('-----------------------------------------------')

        attempts = self.MIP_parameters['attempts_DNN_WDP']
        for attempt in range(1, attempts+1):
            logging.debug('Initialize MIP')
            X = MLCA_NNMIP(DNNs, L=self.MIP_parameters['bigM'])
            if not self.MIP_parameters['mip_bounds_tightening']:
                X.initialize_mip(verbose=False, bidder_specific_constraints=bidder_specific_constraints)
            elif self.MIP_parameters['mip_bounds_tightening'] == 'IA':
                X.tighten_bounds_IA(upper_bound_input=[1]*self.M)
                #X.print_upper_bounds(only_zeros=True)
                X.initialize_mip(verbose=False, bidder_specific_constraints=bidder_specific_constraints)
            elif self.MIP_parameters['mip_bounds_tightening'] == 'LP':
                X.tighten_bounds_IA(upper_bound_input=[1]*self.M)
                X.tighten_bounds_LP(upper_bound_input=[1]*self.M)
                #X.print_upper_bounds(only_zeros=True)
                X.initialize_mip(verbose=False, bidder_specific_constraints=bidder_specific_constraints)
            try:
                logging.info('Solving MIP')
                logging.info('Attempt no: %s', attempt)
                if self.MIP_parameters['warm_start'] and self.warm_start_sol[economy_key] is not None:
                    logging.debug('Using warm start')
                    self.warm_start_sol[economy_key] = X.solve_mip(log_output=False, time_limit=self.MIP_parameters['time_limit'],
                                                                   mip_relative_gap=self.MIP_parameters['relative_gap'], integrality_tol=self.MIP_parameters['integrality_tol'],
                                                                   mip_start=docplex.mp.solution.SolveSolution(X.Mip, self.warm_start_sol[economy_key].as_dict()))
                else:
                    self.warm_start_sol[economy_key] = X.solve_mip(log_output=False, time_limit=self.MIP_parameters['time_limit'],
                                                                   mip_relative_gap=self.MIP_parameters['relative_gap'], integrality_tol=self.MIP_parameters['integrality_tol'])

                if bidder_specific_constraints is None:
                    logging.debug('SET ARGMAX ALLOCATION FOR ALL BIDDERS')
                    b = 0
                    for bidder in self.argmax_allocation[economy_key].keys():
                        self.argmax_allocation[economy_key][bidder][0] = X.x_star[b, :]
                        b = b + 1
                else:
                    logging.debug('SET ARGMAX ALLOCATION ONLY BIDDER SPECIFIC for {}'.format(list(bidder_specific_constraints.keys())[0]))
                    for bidder in bidder_specific_constraints.keys():
                        b = X.get_bidder_key_position(bidder_key=bidder)  # transform bidder_key into bidder position in MIP
                        self.argmax_allocation[economy_key][bidder][1] = X.x_star[b, :]  # now on position 1!

                for key,value in self.argmax_allocation[economy_key].items():
                    logging.debug(key + ':  %s | %s', value[0], value[1])

                self.elapsed_time_mip[economy_key].append(X.soltime)
                break
            except Exception:
                logging.warning('-----------------------------------------------')
                logging.warning('NOT SUCCESSFULLY SOLVED in attempt: %s \n', attempt)
                logging.warning(X.Mip.solve_details)
                if attempt == attempts:
                    X.Mip.export_as_lp(basename='UnsolvedMip_iter{}_{}'.format(self.mlca_iteration, economy_key),path=os.getcwd(), hide_user_names=False)
                    sys.exit('STOP, not solved succesfully in {} attempts\n'.format(attempt))
                clear_session()
                logging.debug('REFITTING:')
                self.estimation_step(economy_key=economy_key)
            DNNs = OrderedDict(list((key, self.NN_models[economy_key][key].model) for key in list(self.NN_models[economy_key].keys())))
        del X
        del DNNs

    def calculate_mlca_allocation(self, economy='Main Economy'):
        logging.info('Calculate MLCA allocation: %s', economy)
        active_bidders = self.economies_names[economy]
        logging.debug('Active bidders: %s', active_bidders)
        allocation, objective = self.solve_WDP(elicited_bids=OrderedDict(list((k, self.elicited_bids.get(k, None)) for k in active_bidders)))
        logging.debug('MLCA allocation in %s:', economy)
        for key,value in allocation.items():
            logging.debug('%s %s', key, value)
        logging.debug('Social Welfare: %s', objective)
        # setting allocations
        if economy == 'Main Economy':
            self.mlca_allocation = allocation
            self.mlca_scw = objective
        if economy in self.mlca_marginal_allocations.keys():
            self.mlca_marginal_allocations[economy] = allocation
            self.mlca_marginal_scws[economy] = objective

    def solve_WDP(self, elicited_bids, verbose=0):  # rem.: objective always rescaled to true values
        bidder_names = list(elicited_bids.keys())
        if verbose==1: logging.debug('Solving WDP based on elicited bids for bidder: %s', bidder_names)
        elicited_bundle_value_pairs = [np.concatenate((bids[0], bids[1].reshape(-1, 1)), axis=1) for bidder, bids in elicited_bids.items()] #transform self.elicited_bids into format for WDP class
        wdp = MLCA_WDP(elicited_bundle_value_pairs)
        wdp.initialize_mip(verbose=0)
        wdp.solve_mip(verbose)
        #TODO: check solution formater
        objective = wdp.Mip.objective_value
        allocation = util.format_solution_mip_new(Mip=wdp.Mip, elicited_bids=elicited_bundle_value_pairs,
                                                     bidder_names=bidder_names, fitted_scaler=self.fitted_scaler)
        if self.fitted_scaler is not None:
            if verbose==1:
                logging.debug('')
                logging.debug('*SCALING*')
                logging.debug('---------------------------------------------')
                logging.debug('WDP objective scaled: %s:', objective)
                logging.debug('WDP objective value scaled by: 1/%s',round(self.fitted_scaler.scale_[0],8))
            objective = float(self.fitted_scaler.inverse_transform([[objective]]))
            if verbose==1:
                logging.debug('WDP objective orig: %s:', objective)
                logging.debug('---------------------------------------------')
        return(allocation, objective)

    def calculate_efficiency_of_allocation(self, allocation, allocation_scw, verbose=0):
        self.solve_SATS_auction_instance()
        efficiency =  allocation_scw/self.SATS_auction_instance_scw
        if verbose==1:
            logging.debug('Calculating efficiency of input allocation:')
            for key,value in allocation.items():
                logging.debug('%s %s', key, value)
            logging.debug('Social Welfare: %s', allocation_scw)
            logging.debug('Efficiency of allocation: %s', efficiency)
        return(efficiency)

    def calculate_vcg_payments(self, forced_recalc=False):
        logging.debug('Calculate payments')

        # (i) solve marginal MIPs
        for economy in list(self.economies_names.keys()):
            if not forced_recalc:
                if economy == 'Main Economy' and self.mlca_allocation is None:
                    self.calculate_mlca_allocation()
                elif economy in self.mlca_marginal_allocations.keys() and self.mlca_marginal_allocations[economy] is None:
                    self.calculate_mlca_allocation(economy=economy)
                else:
                    logging.debug('Allocation for %s already calculated', economy)
            else:
                logging.debug('Forced recalculation of %s', economy)
                self.calculate_mlca_allocation(economy=economy) # Recalc economy

        # (ii) calculate VCG terms for this economy
        for bidder in self.bidder_names:
            marginal_economy_bidder = 'Marginal Economy -({})'.format(util.key_to_int(bidder))
            p1 = self.mlca_marginal_scws[marginal_economy_bidder]  # social welfare of the allocation a^(-i) in this economy
            p2 = sum([self.mlca_allocation[i]['value'] for i in self.economies_names[marginal_economy_bidder]])  # social welfare of mlca allocation without bidder i
            self.mlca_payments[bidder] = round(p1-p2,2)
            logging.info('Payment %s: %s - %s  =  %s', bidder, p1, p2, self.mlca_payments[bidder])
        revenue = sum([self.mlca_payments[i] for i in self.bidder_names])
        logging.info('Revenue: {} | {}% of SCW in efficienct allocation\n'.format(revenue, revenue/self.SATS_auction_instance_scw))
# %%
print('Class MLCA_Economies imported')