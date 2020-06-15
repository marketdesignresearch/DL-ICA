#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file stores helper functions used across the files in this project.

"""

# Libs
import numpy as np
import random
import re
import logging
from collections import OrderedDict

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'

# %% (0) HELPER FUNCTIONS
# %%
def timediff_d_h_m_s(td):
    # can also handle negative datediffs
    if (td).days < 0:
        td = -td
        return -(td.days), -int(td.seconds/3600), -int(td.seconds/60)%60, -(td.seconds%60)
    return td.days, int(td.seconds/3600), int(td.seconds/60)%60, td.seconds%60

# %% Tranforms bidder_key to integer bidder_id
# key = valid bidder_key (string), e.g. 'Bidder_0'
def key_to_int(key):
        return(int(re.findall(r'\d+', key)[0]))

# %% PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS for MLCA MECHANISM
# THIS METHOD USES TRUE UNIFORM SAMPLING!
# SATS_auction_instance = single instance of a value model
# number_initial_bids = number of initial bids
# bidder_ids = bidder ids in this value model (int)
# scaler = scale the y values across all bidders, fit on the selected training set and apply on the validation set


def initial_bids_mlca_unif(SATS_auction_instance, number_initial_bids, bidder_names, scaler=None):
    initial_bids = OrderedDict()
    for bidder in bidder_names:
        logging.debug('Set up intial Bids for: %s', bidder)
        D = unif_random_bids(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_initial_bids)
        null = np.zeros(D.shape[1]).reshape(1, -1) # add null bundle
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids[bidder] = [X, Y]
    if scaler is not None:
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return(initial_bids, scaler)

# %% PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS
# THIS METHOD USES TRUE UNIFORM SAMPLING!
# value_model = single instance of a value model
# c0 = number of initial bids
# bidder_ids = bidder ids in this value model (int)
# scaler = scale the y values across all bidders, fit on the selected training set and apply on the validation set


def initial_bids_pvm_unif(value_model, c0, bidder_ids, scaler=None):
    initial_bids = OrderedDict()
    for bidder_id in bidder_ids:
        logging.debug('Set up intial Bids for: Bidder_{}'.format(bidder_id))
        D = unif_random_bids(value_model=value_model, bidder_id=bidder_id, n=c0)
        # add zero bundle
        null = np.zeros(D.shape[1]).reshape(1, -1)
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids['Bidder_{}'.format(bidder_id)] = [X, Y]
    if scaler is not None:
        tmp = np.array([])
        for bidder_id in bidder_ids:
            tmp = np.concatenate((tmp, initial_bids['Bidder_{}'.format(bidder_id)][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return([initial_bids, scaler])
# %% PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS
# THIS METHOD USES RANDOM SAMPLING OF BUNDLES FROM SATS VIA NORMAL DISTRIBUTION!
# value_model = single instance of a value model
# c0 = number of initial bids
# bidder_ids = bidder ids in this value model (int)
# scaler = scale the y values across all bidders, fit on the selected training set and apply on the validation set
# seed = seed corresponding to the bidder_ids


def initial_bids_pvm(value_model, c0, bidder_ids, scaler=None, seed=None):
    initial_bids = OrderedDict()
    for bidder_id in bidder_ids:
        logging.debug('Set up intial Bids for: Bidder_{}'.format(bidder_id))
        if seed is not None:
            D = np.array(value_model.get_random_bids(bidder_id=bidder_id, number_of_bids=c0, seed=seed[bidder_id]))
        else:
            D = np.array(value_model.get_random_bids(bidder_id=bidder_id, number_of_bids=c0))
        # add zero bundle
        null = np.zeros(D.shape[1]).reshape(1, -1)
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids['Bidder_{}'.format(bidder_id)] = [X, Y]
    if scaler is not None:
        tmp = np.array([])
        for bidder_id in bidder_ids:
            tmp = np.concatenate((tmp, initial_bids['Bidder_{}'.format(bidder_id)][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s', scaler.scale_, ' | ', float(scaler.data_max_ * scaler.scale_), '== feature range max?')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return([initial_bids, scaler])
# %% This function formates the solution of the winner determination problem (WDP) given elicited bids.
# Mip = A solved DOcplex instance.
# elicited_bids = the set of elicited bids for each bidder corresponding to the WDP.
# bidder_names = bidder names (string, e.g., 'Bidder_1')
# fitted_scaler = the fitted scaler used in the valuation model.


def format_solution_mip_new(Mip, elicited_bids, bidder_names, fitted_scaler):
    tmp = {'good_ids': [], 'value': 0}
    Z = OrderedDict()
    for bidder_name in bidder_names:
        Z[bidder_name] = tmp
    S = Mip.solution.as_dict()
    for key in list(S.keys()):
        index = [int(x) for x in re.findall(r'\d+', key)]
        bundle = elicited_bids[index[0]][index[1], :-1]
        value = elicited_bids[index[0]][index[1], -1]
        if fitted_scaler is not None:
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug(value)
            logging.debug('WDP values for allocation scaled by: 1/%s',round(fitted_scaler.scale_[0],8))
            value = float(fitted_scaler.inverse_transform([[value]]))
            logging.debug(value)
            logging.debug('---------------------------------------------')
        bidder = bidder_names[index[0]]
        Z[bidder] = {'good_ids': list(np.where(bundle == 1)[0]), 'value': value}
    return(Z)
# %% This function generates bundle-value pairs for a single bidder sampled uniformly at random from the bundle space.
# value_model = SATS auction model instance generated via PySats
# bidder_id = bidder id (int)
# n = number of bundle-value pairs.


def unif_random_bids(value_model, bidder_id, n):
    logging.debug('Sampling uniformly at random %s bundle-value pairs from bidder %s',n, bidder_id)
    ncol = len(value_model.get_good_ids())  # number of items in value model
    D = np.unique(np.asarray(random.choices([0,1], k=n*ncol)).reshape(n,ncol), axis=0)
    # get unique ones if accidently sampled equal bundle
    while D.shape[0] != n:
        tmp = np.asarray(random.choices([0,1], k=ncol)).reshape(1, -1)
        D = np.unique(np.vstack((D, tmp)), axis=0)
    # define helper function for specific bidder_id
    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)
    D = np.hstack((D, np.apply_along_axis(myfunc, 1, D).reshape(-1, 1)))
    del myfunc
    return(D)
