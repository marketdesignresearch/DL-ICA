# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:41:49 2020

@author: jakob
"""

"""
FILE DESCRIPTION:

This file contains the main function mlca() for running the DNN-based MLCA mechanism as described in Algorithm 3 & 4, Machine Learning-powered Iterative Combinatorial Auctions, Working Paper Brero et al., 2019

The function mlca() runs the DNN-based MLCA-Mechanism and outputs the MLCA-allocation, the MLCA-payments, the elicited bids, and further statistics on the MLCA run.
The arguments are given as follows:
    SATS_domain_name = name of the Spectrum Auction Test Suite (SATS) domain: 'GSVM', 'LSVM', 'MRVM'
    SATS_auction_instance_seed = seed for the auction instance generated via PySats.py
    Qinit = number of initial bundle-value pairs (bids) per bidder
    Qmax = maximum number of value queries (including Qinit) per bidder
    Qround = number of sampled marginals per bidder per auctin round == number of value queries per round
    NN_parameters = Neural Network parameters (used here tensorflow.keras)
    MIP_parameters = Mixed Integer Program parameters (udes here docplex.mp)
    scaler = scaler instance (sklearn.preprocessing) used for scaling the values of the initially sampled bundle-value pairs
    init_bids_and_fitted_scaler = self specified initial bids per bidder and cporresponding fitted_scaler
    return_allocation = boolean, should the MLCA-allocation be calculated and returned
    return_payments = boolean, should the MLCA-payments be calculated and returned
    calc_efficiency_per_iteration = boolean, should the efficient allocation given elicited bids be calculated per auction round

See example_mlca.py for an example of how to use the function MLCA.

"""

# Libs
from datetime import datetime
import logging
import random
from tensorflow.keras.backend import clear_session

# Own Modules
from source.pysats import PySats
from source.mlca.mlca_economies import MLCA_Economies
from source.mlca.mlca_util import key_to_int, timediff_d_h_m_s

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'

#%% MLCA MECHANISM SINGLE RUN
def mlca_mechanism(SATS_domain_name, SATS_auction_instance_seed, Qinit, Qmax, Qround, NN_parameters, MIP_parameters, scaler,
                   init_bids_and_fitted_scaler=[None,None], return_allocation=False, return_payments=False, calc_efficiency_per_iteration=False):

    start = datetime.now()
    logging.warning('START MLCA:')
    logging.warning('-----------------------------------------------')
    OUTPUT={}
    if not scaler:
        scaler = None
    logging.warning('Model: %s',SATS_domain_name)
    logging.warning('Seed SATS Instance: %s',SATS_auction_instance_seed)
    logging.warning('Qinit: %s',Qinit)
    logging.warning('Qmax: %s',Qmax)
    logging.warning('Qround: %s\n',Qround)

    random.seed(SATS_auction_instance_seed)  # (*) truly uniform bids: global seed

    # Instantiate Economies
    logging.debug('Instantiate SATS Instance')
    if SATS_domain_name == 'LSVM':
        SATS_auction_instance = PySats.getInstance().create_lsvm(seed=SATS_auction_instance_seed)  # create SATS auction instance
    if SATS_domain_name == 'GSVM':
        SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_auction_instance_seed)  # create SATS auction instance
    if SATS_domain_name == 'MRVM':
        SATS_auction_instance = PySats.getInstance().create_mrvm(seed=SATS_auction_instance_seed)  # create SATS auction instance
    E = MLCA_Economies(SATS_auction_instance=SATS_auction_instance, SATS_auction_instance_seed=SATS_auction_instance_seed,
                       Qinit=Qinit, Qmax=Qmax, Qround=Qround, scaler=scaler)  # create economy instance

    E.set_NN_parameters(parameters=NN_parameters)   # set NN parameters
    E.set_MIP_parameters(parameters=MIP_parameters)   # set MIP parameters

    # Set initial bids | Line 1-3
    init_bids, init_fitted_scaler = init_bids_and_fitted_scaler
    if init_bids is not None:
        E.set_initial_bids(initial_bids=init_bids, fitted_scaler=init_fitted_scaler) # (*) use self defined inital bids | Line 1
    else:
        E.set_initial_bids()  # (*) create inital bids B^0, uniformly sampling at random from bundle space | Line 2

    # Calculate efficient allocation given current elicited bids
    if calc_efficiency_per_iteration: E.calculate_efficiency_per_iteration()

    # Global while loop: check if for all bidders one addtitional auction round is feasible | Line 4
    Rmax = max(E.get_number_of_elicited_bids().values())
    CHECK = Rmax <= (E.Qmax-E.Qround)
    while CHECK:

        # Increment iteration
        E.mlca_iteration+=1
        # log info
        E.get_info()
        # Check Current Query Profile: | Line 5
        logging.debug('Current query profile S=(S_1,...,S_n):')
        for k,v in E.current_query_profile.items():
            logging.debug(k+':  %s',v)

        # Marginal Economies: | Line 6-12
        logging.info('MARGINAL ECONOMIES FOR ALL BIDDERS')
        logging.info('-----------------------------------------------\n')
        for bidder in E.bidder_names:
            logging.info(bidder)
            logging.info('-----------------------------------------------')
            logging.debug('Sampling marginals for %s', bidder)
            sampled_marginal_economies = E.sample_marginal_economies(active_bidder=key_to_int(bidder), number_of_marginals=E.Qround-1)
            sampled_marginal_economies.sort()
            logging.debug('Calculate next queries for the following sampled marginals:')
            for marginal_economy in sampled_marginal_economies:
                logging.debug(marginal_economy)

            for marginal_economy in sampled_marginal_economies:
                logging.debug('')
                logging.info(bidder + ' | '+ marginal_economy)
                logging.info('-----------------------------------------------')
                logging.info('Status of Economy: %s\n', E.economy_status[marginal_economy])
                q_i = E.next_queries(economy_key=marginal_economy, active_bidder=bidder)
                E.update_current_query_profile(bidder=bidder, bundle_to_add=q_i)
                logging.debug('')
                logging.debug('Current query profile for %s:', bidder)
                for k in range(E.current_query_profile[bidder].shape[0]):
                    logging.debug(E.current_query_profile[bidder][k,:])
                logging.debug('')
        # Main Economy: | Line 13-14
        logging.info('MAIN ECONOMY FOR ALL BIDDERS')
        logging.info('-----------------------------------------------\n')
        for bidder in E.bidder_names:
            logging.info(bidder)
            logging.info('-----------------------------------------------')
            economy_key = 'Main Economy'
            logging.debug(economy_key)
            logging.debug('-----------------------------------------------')
            logging.debug('Status of Economy: %s', E.economy_status[economy_key])
            q_i = E.next_queries(economy_key=economy_key, active_bidder=bidder)
            E.update_current_query_profile(bidder=bidder, bundle_to_add=q_i)
            logging.debug('')
            logging.debug('Current query profile for %s:', bidder)
            for bundle in range(E.current_query_profile[bidder].shape[0]):
                logging.debug(E.current_query_profile[bidder][k,:])
            logging.debug('')

        # Update Elicited Bids With Current Query Profile and check uniqueness | Line 15-16
        if not E.update_elicited_bids(): return('UNIQUENESS CHECK FAILED see logfile')

        # Reset Attributes | Line 18
        logging.info('RESET: Auction Round Query Profile S=(S_1,...,S_n)')
        E.reset_current_query_profile()
        logging.info('RESET: Status of Economies')
        E.reset_economy_status()
        logging.info('RESET: NN Models')
        E.reset_NN_models()
        logging.info('RESET: Argmax Allocation\n')
        E.reset_argmax_allocations()
        clear_session()  # clear keras session

        # Calculate efficient allocation given current elicited bids
        if calc_efficiency_per_iteration: E.calculate_efficiency_per_iteration()

        # Update while condition
        Rmax = max(E.get_number_of_elicited_bids().values())
        CHECK = Rmax <= (E.Qmax-E.Qround)

    # allocation & payments # | Line 20
    if return_allocation:
        logging.info('ALLOCATION')
        logging.info('---------------------------------------------')
        E.calculate_mlca_allocation()
        E.mlca_allocation_efficiency = E.calculate_efficiency_of_allocation(E.mlca_allocation, E.mlca_scw, verbose=1)
    if return_payments: # | Line 21
        logging.info('')
        logging.info('PAYMENTS')
        logging.info('---------------------------------------------')
        E.calculate_vcg_payments()

    end = datetime.now()
    total_time_elapsed = '{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(end-start))
    E.get_info(final_summary=True)
    logging.warning('EFFICIENCY: {} %'.format(round(E.mlca_allocation_efficiency,4)*100))
    logging.warning('TOTAL TIME ELAPSED: {}'.format(total_time_elapsed))
    logging.warning('MLCA FINISHED')

    # Set OUTPUT
    OUTPUT['MLCA Efficiency'] = E.mlca_allocation_efficiency
    OUTPUT['MLCA Allocation'] = E.mlca_allocation
    OUTPUT['MLCA Payments'] = E.mlca_payments
    OUTPUT['Statistics']={'Total Time Elapsed':total_time_elapsed, 'Elapsed Times of MIPs': E.elapsed_time_mip,
                          'NN Losses': E.losses, 'Efficiency per Iteration': E.efficiency_per_iteration,
                          'Efficient allocation per Iteration': E.efficient_allocation_per_iteration}
    OUTPUT['Elicited Bids'] = E.elicited_bids

    return(SATS_auction_instance_seed, OUTPUT, E)
#%%
print('MLCA function imported')