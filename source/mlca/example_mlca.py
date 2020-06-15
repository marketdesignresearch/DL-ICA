#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file presents an example of how to run the MLCA mechanism wit the function mlca() from the file mlca.py

"""

# Libs
import logging
import pandas as pd
from collections import OrderedDict

# Own Modules
from source.mlca.mlca import mlca_mechanism

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'
#%% define logger

#clear existing logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# log debug to console
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S')
logging.basicConfig(level=logging.INFO, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S')
#logging.basicConfig(level=logging.WARNING, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S')
#%% Define parameters

#TODO test mlca, on cluster


# (1) Value model parameters

# =============================================================================
# SATS_domain_name = 'LSVM'
# N = 6  # number of bidders
# M = 18  # number of items
# bidder_types = 2
# bidder_ids = list(range(0, 6))
# scaler = False
# print('\n------------------------ SATS parameters ------------------------')
# print('Value Model:', SATS_domain_name)
# print('Number of Bidders: ', N)
# print('Number of BidderTypes: ', bidder_types)
# print('Number of Items: ', M)
# print('Scaler: ', scaler)
# =============================================================================

# GSVM
SATS_domain_name = 'GSVM'
N = 7  # number of bidders
M = 18  # number of items
bidder_types = 2
bidder_ids = list(range(0, 7))
scaler = False
print('\n------------------------ SATS parameters ------------------------')
print('Value Model: ', SATS_domain_name)
print('Number of Bidders: ', N)
print('Number of BidderTypes: ', bidder_types)
print('Number of Items: ', M)
print('Scaler: ', scaler)

# =============================================================================
#
# # MRVM
# SATS_domain_name = 'MRVM'
# N = 10  # number of bidders
# M = 98  # number of items
# bidder_types = 3
# bidder_ids = list(range(0, 10))
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 500))
# print('\n------------------------ SATS parameters ------------------------')
# print('\nValue Model: ', SATS_domain_name)
# print('Number of Bidders: ', N)
# print('Number of BidderTypes: ', bidder_types)
# print('Number of Items: ', M)
# print('Scaler: ', scaler)
# =============================================================================

# (2) Neural Network Parameters
epochs = 200
batch_size = 30
regularization_type = 'l2'  # 'l1', 'l2' or 'l1_l2'
# national bidder LSVM: id=0, GSVM:id=6, MRVM:id=7,8,9
regularization_N = 1e-5
learning_rate_N = 0.01
layer_N = [10, 10]
dropout_N = True
dropout_prob_N = 0.05
# regional bidders LSVM: id=1-5, GSVM:id=0-5, MRVM:id=3,4,5,6
regularization_R = 1e-5
learning_rate_R = 0.01
layer_R = [32, 32]
dropout_R = True
dropout_prob_R = 0.05
# local bidders MRVM:id=0,1,2
regularization_L = 1e-5
learning_rate_L = 0.01
layer_L = [10, 10]
dropout_L = True
dropout_prob_L = 0.05
NN_parameters = {}
if SATS_domain_name == 'LSVM':
    for bidder_id in bidder_ids:
        if bidder_id == 0:
            NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization',regularization_N),
                                                                         ('learning_rate',learning_rate_N), ('architecture',layer_N),
                                                                         ('dropout', dropout_N), ('dropout_prob', dropout_prob_N),
                                                                         ('epochs', epochs), ('batch_size',batch_size),
                                                                         ('regularization_type', regularization_type)])
        else:
            NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization',regularization_R),
                                                                         ('learning_rate',learning_rate_R), ('architecture',layer_R),
                                                                         ('dropout', dropout_R), ('dropout_prob', dropout_prob_R),
                                                                         ('epochs', epochs), ('batch_size',batch_size),
                                                                         ('regularization_type', regularization_type)])
if SATS_domain_name == 'GSVM':
    for bidder_id in bidder_ids:
        if bidder_id == 6:
            NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization',regularization_N),
                                                                         ('learning_rate',learning_rate_N), ('architecture',layer_N),
                                                                         ('dropout', dropout_N), ('dropout_prob', dropout_prob_N),
                                                                         ('epochs', epochs), ('batch_size',batch_size),
                                                                         ('regularization_type', regularization_type)])
        else:
            NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization',regularization_R),
                                                                         ('learning_rate',learning_rate_R), ('architecture',layer_R),
                                                                         ('dropout', dropout_R), ('dropout_prob', dropout_prob_R),
                                                                         ('epochs', epochs), ('batch_size',batch_size),
                                                                         ('regularization_type', regularization_type)])
if SATS_domain_name == 'MRVM':
    for bidder_id in bidder_ids:
        if bidder_id in [0, 1, 2]:
            NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization',regularization_L),
                                                                         ('learning_rate',learning_rate_L), ('architecture',layer_L),
                                                                         ('dropout', dropout_L), ('dropout_prob', dropout_prob_L),
                                                                         ('epochs', epochs), ('batch_size',batch_size),
                                                                         ('regularization_type', regularization_type)])
        if bidder_id in [3, 4, 5, 6]:
            NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization',regularization_R),
                                                                         ('learning_rate',learning_rate_R), ('architecture',layer_R),
                                                                         ('dropout', dropout_R), ('dropout_prob', dropout_prob_R),
                                                                         ('epochs', epochs), ('batch_size',batch_size),
                                                                         ('regularization_type', regularization_type)])
        if bidder_id in [7, 8, 9]:
            NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization',regularization_N),
                                                                         ('learning_rate',learning_rate_N), ('architecture',layer_N),
                                                                         ('dropout', dropout_N), ('dropout_prob', dropout_prob_N),
                                                                         ('epochs', epochs), ('batch_size',batch_size),
                                                                         ('regularization_type', regularization_type)])


print('\n------------------------ DNN  parameters ------------------------')
print('Epochs:', epochs)
print('Batch Size:', batch_size)
print('Regularization:', regularization_type)
for key in list(NN_parameters.keys()):
    print()
    print(key+':')
    [print(k+':',v) for k,v in NN_parameters[key].items()]


# (3) MIP parameters
bigM = 2000
Mip_bounds_tightening = 'IA'   # False ,'IA' or 'LP'
warm_start = False
time_limit = 1800  #1h = 3600sec
relative_gap = 0.001
integrality_tol = 1e-8
attempts_DNN_WDP = 5
MIP_parameters = OrderedDict([('bigM',bigM),('mip_bounds_tightening',Mip_bounds_tightening), ('warm_start',warm_start),
                              ('time_limit',time_limit), ('relative_gap',relative_gap), ('integrality_tol',integrality_tol), ('attempts_DNN_WDP',attempts_DNN_WDP)])
print('\n------------------------ MIP  parameters ------------------------')
for key,v in MIP_parameters.items():
    print(key+':', v)

# (4) MLCA specific parameters
Qinit = 30
Qmax = 58
Qround = N
SATS_auction_instance_seed = 10

print('\n------------------------ MLCA  parameters ------------------------')
print('Qinit:', Qinit)
print('Qmax:', Qmax)
print('Qround:', Qround)
print('Seed SATS Instance: ', SATS_auction_instance_seed)
#%% Start DNN-based MLCA

res = mlca_mechanism(SATS_domain_name=SATS_domain_name, SATS_auction_instance_seed=SATS_auction_instance_seed, Qinit=Qinit, Qmax=Qmax,
                        Qround=Qround, NN_parameters=NN_parameters, MIP_parameters=MIP_parameters, scaler=scaler, init_bids_and_fitted_scaler=[None,None],
                        return_allocation=True, return_payments=True, calc_efficiency_per_iteration=True)
# %% Continue with elicited bids from above
B = res[1]['Elicited Bids']
res2 = mlca_mechanism(SATS_domain_name=SATS_domain_name, SATS_auction_instance_seed=SATS_auction_instance_seed, Qinit=Qinit, Qmax=Qmax+Qround,
                        Qround=Qround, NN_parameters=NN_parameters, MIP_parameters=MIP_parameters, scaler=scaler, init_bids_and_fitted_scaler=[B,None],
                        return_allocation=True, return_payments=True, calc_efficiency_per_iteration=True)
#%% Result analysis
# TODO: code result analysis
# prepare results
EFF = RESULT[1][0]
ALLOCS = RESULT[1][1]
PAYMENTS = RESULT[1][2]
TIME_STATS = RESULT[1][3]
ITERATIONS = RESULT[1][4]
BOUNDS = RESULT[1][5]
MIP_STATS_TIME = RESULT[1][6]
mip_time = []
for k, v in MIP_STATS_TIME.items():
    mip_time = mip_time + v
Miptimes = pd.DataFrame(mip_time)
ACTUAL_QUERIES = RESULT[1][7]
Actualqueries = pd.DataFrame.from_dict(ACTUAL_QUERIES, orient='index')
Allocation_sats = pd.DataFrame.from_dict(ALLOCS[0]).transpose()
Allocation_sats = Allocation_sats.rename(index={k: 'Bidder_{}'.format(k) for k in range(0, N)})
sum_sats = sum(Allocation_sats['value'])
Allocation_pea = pd.DataFrame.from_dict(ALLOCS[1]).transpose()
Allocation_pea = Allocation_pea.rename(index={'Bidder_{}'.format(k): 'Bidder_{}'.format(k) for k in range(0, N)})
sum_pvm = sum(Allocation_pea['value'])
payments = pd.DataFrame.from_dict(PAYMENTS,  orient='index')
Rev_pos = payments[payments > 0].sum(axis=0).values[0]
Rev = payments.sum(axis=0).values[0]
RelRev_pos = Rev_pos/sum_sats*100
RelRev = Rev/sum_sats*100
total_bounds = pd.DataFrame.from_dict(BOUNDS, orient='index').describe()

# print results
print('---------------------------------  RESULT ---------------------------------------')
print('Valuation Model:', SATS_domain_name)
print('Instance seed:', RESULT[0])
print('---------------------------------------------------------------------------------')
print('Efficiency in %: ', EFF)
print()
print('Payments:')
for k, v in PAYMENTS.items():
    print(k, ':', v)
print()
print('Relative Revenue in %: ', RelRev)
print('Relative Revenue (if payments capped at zero) in %: ', RelRev_pos)
print('\nIterations:')
for k, v in ITERATIONS.items():
    print(k, ':', v)
print()
print('Instance Time (hh:mm:ss): ' + TIME_STATS)
print('Average MIP Time: {} sec'.format(round(Miptimes.describe().loc['mean'].mean(), 4)))
print('Median MIP Time: {} sec'.format(round(Miptimes.describe().loc['50%'].mean(), 4)))
print('Average #Queries per bidder - c0: {}'.format(Actualqueries.mean(axis=0).values[0]))
print('Average Maximium #Queries per bidder - c0: {}'.format(Actualqueries.max(axis=0).values[0]))
print('---------------------------------------------------------------------------------')
if SATS_domain_name == 'LSVM':
    Na = ['Bidder_0']
    Re = ['Bidder_{}'.format(k) for k in range(1, 6)]
if SATS_domain_name == 'GSVM':
    Na = ['Bidder_6']
    Re = ['Bidder_{}'.format(k) for k in range(0, 6)]
if SATS_domain_name == 'MRVM':
    Na = ['Bidder_7', 'Bidder_8', 'Bidder_9']
    Re = ['Bidder_3', 'Bidder_4', 'Bidder_5', 'Bidder_6']
    Lo = ['Bidder_0', 'Bidder_1', 'Bidder_2']

print('SATS Social Welfare:', round(sum_sats, 2))
print('Allocation:')
print(Allocation_sats['good_ids'].to_string())
print('\nValues:')
if SATS_domain_name == 'MRVM':
    print('Local bidders:')
    print(Allocation_sats['value'][Lo].to_string())
    ALocal = Allocation_sats['value'][Lo].sum()/sum_sats
    ALocal = round(ALocal*100, 2)
    print('% of SATS Social Welfare', ALocal, '\n')
print('Regional bidders:')
print(Allocation_sats['value'][Re].to_string())
AReg = Allocation_sats['value'][Re].sum()/sum_sats
AReg = round(AReg*100, 2)
print('% of SATS Social Welfare', AReg, '\n')
print('National bidders:')
print(Allocation_sats['value'][Na].to_string())
ANat = Allocation_sats['value'][Na].sum()/sum_sats
ANat = round(ANat*100, 2)
print('% of SATS Social Welfare', ANat)
print('---------------------------------------------------------------------------------')
print('PVM Social Welfare:', round(sum_pvm, 2))
print('Allocation:')
print(Allocation_pea['good_ids'].to_string())
print('\nValues:')
if SATS_domain_name == 'MRVM':
    print('Local bidders:')
    print(Allocation_pea['value'][Lo].to_string())
    ALocal = Allocation_pea['value'][Lo].sum()/sum_sats
    ALocal = round(ALocal*100, 2)
    print('% of SATS Social Welfare', ALocal, '\n')
print('Regional bidders:')
print(Allocation_pea['value'][Re].to_string())
AReg = Allocation_pea['value'][Re].sum()/sum_sats
AReg = round(AReg*100, 2)
print('% of SATS Social Welfare', AReg, '\n')
print('National bidders:')
print(Allocation_pea['value'][Na].to_string())
ANat = Allocation_pea['value'][Na].sum()/sum_sats
ANat = round(ANat*100, 2)
print('% of SATS Social Welfare', ANat)
print('---------------------------------------------------------------------------------')
