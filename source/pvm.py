#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file contains the main function PVM() for running the DNN-based PVM mechanism as described in Algorithm 2, Section 2 of the AAAI 2020 paper Deep Learning-powered Iterative Combinatorial Auctions.

The function PVM() runs the DNN-based Pseudo-VCG-Mechanism and outputs the efficiency of the obtained allocation, the allocation and payments, and further statistics on the PVM run. The arguments are given as follows:
    scaler = scaler instance used for scaling the values
    caps = [c0,ce] caps for the preference elicitation algorithm (PEA)
    L = global 'big-M' constraint
    parameters = DNN parameters
    epochs = epochs in training the DNNs
    batch_size = batch size for trainng the DNNs
    model_name = valuation model name (string) can be 'GSVM', 'LSVM' or 'MRVM'
    sample_weight_on = boolean, if weights are used for elicited bundles in the next traininn step of the DNNs (not used in AAAI 2020 paper)
    sample_weight_scaling = weightvector for each bidder addtionally for scaling the correction weight, i.e., MAE/sample_weight_scaling[i] (see Class_Economies_github.py for an explanation)
    min_iteration = minimum number of iterations for the PEA in each economy (set to 1 in AAAI 2020 paper, hence no restriction)
    seed_instance = seed for the auction instance generated via PySats.py
    regularization_type = regularization for training DNNs (string). Can be 'l1', 'l2' or 'l1_l2'.
    Mip_bounds_tightening = procedure for bound tightening in MIP reformulatio of the DNN-based WDP. Can be 'IA' (Intervall Arithmetic), 'LP' (LP-Reölaxations) or 'False'
    warm_start = boolean determines if a solution to the MIP  of the previous iteration in the PEA should be used as a warm start for the MIP formulation of the DNN-based WDP.

See example_pvm.py for an example of how to use the function PVM.

"""

# Libs
import time
import logging
import random
from keras import backend as K

# Own Modules
from source.pysats import PySats
from source.economies import Economies

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'
# %% PVM MECHANISM SINGLE RUN


def pvm(scaler, caps, L, parameters, epochs, batch_size, model_name, sample_weight_on, sample_weight_scaling, min_iteration, seed_instance, regularization_type, Mip_bounds_tightening, warm_start):
    start0 = time.time()
    if not scaler:
        scaler = None
    c0 = caps[0]
    ce = caps[1]
    logging.debug(seed_instance)
    seeds_random_bids = None  # (*) truly uniform bids: no bidder specific seeds
    random.seed(seed_instance)  # (*) truly uniform bids: global seed

    # (1) PVM Algorithm: START
    logging.debug('\n(1) START PVM:')
    logging.debug('-----------------------------------------------')
    # Instantiate Economies
    if model_name == 'LSVM':
        value_model = PySats.getInstance().create_lsvm(seed=seed_instance)  # create SATS value model instance
    if model_name == 'GSVM':
        value_model = PySats.getInstance().create_gsvm(seed=seed_instance)  # create SATS value model instance
    if model_name == 'MRVM':
        value_model = PySats.getInstance().create_mrvm(seed=seed_instance)  # create SATS value model instance
    # ============================================================================= (instead of (*))
    # # for bids per bidder type when SATS RANDOM SAMPLING is used (not used in AAAI 2020 paper)
    # seeds_random_bids = [list(range(i, i+6)) for i in range(1, 1+number_of_instances*len(value_model.get_bidder_ids()), len(value_model.get_bidder_ids()))][m]
    # =============================================================================

    E = Economies(value_model=value_model, c0=c0, ce=ce, min_iteration=min_iteration, epochs=epochs, batch_size=batch_size, L=L, regularization_type=regularization_type,
                  Mip_bounds_tightening=Mip_bounds_tightening, warm_start=warm_start, scaler=scaler)  # create economy instance
    E.set_initial_bids(seeds_random_bids=seeds_random_bids)  # create inital bids V^0, uniformly sampling at random from bundle space
    E.set_NN_parameters(parameters=parameters)   # set NN parameters
    # activate weights (not used in AAAI 2020 paper)
    if sample_weight_on:
        E.activate_weights(sample_weight_scaling=sample_weight_scaling)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------- #
    while any(list(E.status.values())):
        # ---------------------------------------------------------------------------------------------------------------------------------------------------- #
        for economy_key, status in E.status.items():
            logging.debug(' ')
            logging.debug('(1.0) INFO')
            logging.debug('-----------------------------------------------')
            logging.debug('Instance %s', seed_instance)
            logging.debug('Efficiency Stats:')
            E.info(economy_key=economy_key)
            # check validity of economy
            if status is False:
                logging.debug('ELICITATION FINISHED FOR ECONOMY: {}'.format(economy_key) + '\n')
            else:
                E.calculate_argmax_allocation(economy_key=economy_key)
                # E.do_query(economy_key=economy_key, allocation=E.argmax_allocation)  # version 1 with total caps (not used in AAAI 2020 paper)
                E.do_query_2(economy_key=economy_key, allocation=E.argmax_allocation)  # version 2 with individual caps
                K.clear_session()  # clear keras session
        # ---------------------------------------------------------------------------------------------------------------------------------------------------- #
        # logging.debug('Update Main Economy')
        # E.update_main_economy()  # update bids for main_economy additionally with all the elicited bids from the others from the SAME iteration, ''add marginals version'' (not used in AAAI 2020 paper)
        # reset attributes
        logging.debug('Reset NN Models')
        E.reset_keras_models()
        logging.debug('Reset Argmax allocations')
        E.reset_argmax_allocations()
    # ---------------------------------------------------------------------------------------------------------------------------------------------------- #
    # logging.debug('Update Main Economy')
    # E.add_all_bids_to_main_economy()  # add all bids to main economy, NEW ALLOCATION RULE x^(-\empty set) will not be calculated here; main economy allocation is then x^pvm (not used in AAAI 2020 paper)
    E.info(None)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------- #
    # END of PVM mechanism

    # (2) Calculate Efficiency: START
    logging.debug('\n(2) CALCULATE EFFICIENCY FOR INSTANCE {}'.format(seed_instance))
    logging.debug('-----------------------------------------------')
    logging.debug('Solving WDP SATS:')
    start = time.time()
    allocation_SATS, value_SATS = value_model.get_efficient_allocation()
    end = time.time()
    logging.debug('WDP SATS solved in ' + time.strftime('%H:%M:%S', time.gmtime(end-start)) + '\n')
    logging.debug('Optimal SATS Allocation:')
    for key in list(allocation_SATS.keys()):
        logging.debug('%s: %s', key, allocation_SATS[key])
    logging.debug('Optimal SATS Value: {}'.format(value_SATS) + '\n')
    E.solve_WDP()  # calculate optimal allocations for all economies (x^(-i),v(x^(-i)))
    E.calculate_finalpvm_alloc()  # determine pvm allocation as maximum over allocation of economies
    E.calculate_payments()
    eff = E.pvm_allocation[1]/value_SATS*100
    alloc = [allocation_SATS, E.pvm_allocation[0]]
    end0 = time.time()
    T = time.strftime('%H:%M:%S', time.gmtime(end0-start0))
    logging.info('INSTANCE: {} | EFF: {} % | TIME ELAPSED: {}'.format(seed_instance, round(eff, 3), T))
    # CALCULATE EFFICIENCY: END

    RESULT = [eff, alloc, E.payments, T, E.iteration, E.total_bounds, E.elapsed_time_mip, E.actual_queries_elicitation]
    return((seed_instance, RESULT))


print('PVM function imported')
