#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file presents an example of how to run the PVM mechanism wit the function pvm() from the file pvm.py

"""

# Libs
import logging
from sklearn.preprocessing import MinMaxScaler  # used only for MRVM
import pandas as pd

# Own Modules
from source.pvm import pvm

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
# %% Define parameters

# (1) Value model parameters

model_name = 'LSVM'
N = 6  # number of bidders
M = 18  # number of items
bidder_types = 2
bidder_ids = list(range(0, 6))
scaler = False
print('\n------------------------ SATS parameters ------------------------')
print('Value Model:', model_name)
print('Number of Bidders: ', N)
print('Number of BidderTypes: ', bidder_types)
print('Number of Items: ', M)
print('Scaler: ', scaler)

# =============================================================================
# # GSVM
# model_name = 'GSVM'
# N = 7  # number of bidders
# M = 18  # number of items
# bidder_types = 2
# bidder_ids = list(range(0, 7))
# scaler = False
# print('\n------------------------ INFO  PVM RUN ------------------------')
# print('Value Model: ', model_name)
# print('Number of Bidders: ', N)
# print('Number of BidderTypes: ', bidder_types)
# print('Number of Items: ', M)
# print('Scaler: ', scaler)
# =============================================================================

# =============================================================================
# # MRVM
# model_name = 'MRVM'
# N = 10  # number of bidders
# M = 98  # number of items
# bidder_types = 3
# bidder_ids = list(range(0, 10))
# scaler = MinMaxScaler(feature_range=(0, 500))
# print('\nValue Model: ', model_name)
# print('Number of Bidders: ', N)
# print('Number of BidderTypes: ', bidder_types)
# print('Number of Items: ', M)
# print('Scaler: ', scaler)
# =============================================================================

# (2) Neural Network Parameters
epochs = 300
batch_size = 32
regularization_type = 'l1_l2'  # 'l1', 'l2' or 'l1_l2'
# national bidder LSVM: id=0, GSVM:id=6, MRVM:id=7,8,9
regularization_N = 0.00001
learning_rate_N = 0.01
layer_N = [16, 16]
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
layer_L = [10, 10]
dropout_L = True
dropout_prob_L = 0.05
DNN_parameters = {}
if model_name == 'LSVM':
    for bidder_id in bidder_ids:
        if bidder_id == 0:
            DNN_parameters['Bidder_{}'.format(bidder_id)] = (regularization_N, learning_rate_N, layer_N, dropout_N, dropout_prob_N)
        else:
            DNN_parameters['Bidder_{}'.format(bidder_id)] = (regularization_R, learning_rate_R, layer_R, dropout_R, dropout_prob_R)
if model_name == 'GSVM':
    for bidder_id in bidder_ids:
        if bidder_id == 6:
            DNN_parameters['Bidder_{}'.format(bidder_id)] = (regularization_N, learning_rate_N, layer_N, dropout_N, dropout_prob_N)
        else:
            DNN_parameters['Bidder_{}'.format(bidder_id)] = (regularization_R, learning_rate_R, layer_R, dropout_R, dropout_prob_R)
if model_name == 'MRVM':
    for bidder_id in bidder_ids:
        if bidder_id in [0, 1, 2]:
            DNN_parameters['Bidder_{}'.format(bidder_id)] = (regularization_L, learning_rate_L, layer_L, dropout_L, dropout_prob_L)
        if bidder_id in [3, 4, 5, 6]:
            DNN_parameters['Bidder_{}'.format(bidder_id)] = (regularization_R, learning_rate_R, layer_R, dropout_R, dropout_prob_R)
        if bidder_id in [7, 8, 9]:
            DNN_parameters['Bidder_{}'.format(bidder_id)] = (regularization_N, learning_rate_N, layer_N, dropout_N, dropout_prob_N)
sample_weight_on = False
sample_weight_scaling = None
# =============================================================================
# sample_weight_on = True
# #sample_weight_scaling = [10, 10, 10, 10, 10, 30]  # example for GSVM
# sample_weight_scaling = [30, 10, 10, 10, 10, 10]  # example for LSVM
# #sample_weight_scaling = [10, 10, 10, 10, 10, 10, 10, 30, 30, 30]  # example for MRVM
# =============================================================================

print('\n------------------------ DNN  parameters ------------------------')
print('Epochs:', epochs)
print('Batch Size:', batch_size)
print('Regularization:', regularization_type)
for key in list(DNN_parameters.keys()):
    print(key, DNN_parameters[key])
print('Sample weighting:', sample_weight_on)
print('Sample weight scaling:', sample_weight_scaling)

# (3) MIP parameters
L = 3000
Mip_bounds_tightening = 'IA'   # False ,'IA' or 'LP'
warm_start = False
print('\n------------------------ MIP  parameters ------------------------')
print('Mip_bounds_tightening:', Mip_bounds_tightening)
print('Warm_start:', warm_start)

# (4) PVM specific parameters
caps = [40, 10]  # [c_0, c_e] with initial bids c0 and maximal number of value queries ce
seed_instance = 12
min_iteration = 1
print('\n------------------------ PVM  parameters ------------------------')
print('c0:', caps[0])
print('ce:', caps[1])
print('Seed: ', seed_instance)
print('min_iteration:', min_iteration)
# %% Start DNN-based PVM

RESULT = pvm(scaler=scaler, caps=caps, L=L, parameters=DNN_parameters, epochs=epochs, batch_size=batch_size, model_name=model_name, sample_weight_on=sample_weight_on,
             sample_weight_scaling=sample_weight_scaling, min_iteration=min_iteration, seed_instance=seed_instance, regularization_type=regularization_type,
             Mip_bounds_tightening=Mip_bounds_tightening, warm_start=warm_start)


# %% Result analysis

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
print('Valuation Model:', model_name)
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
if model_name == 'LSVM':
    Na = ['Bidder_0']
    Re = ['Bidder_{}'.format(k) for k in range(1, 6)]
if model_name == 'GSVM':
    Na = ['Bidder_6']
    Re = ['Bidder_{}'.format(k) for k in range(0, 6)]
if model_name == 'MRVM':
    Na = ['Bidder_7', 'Bidder_8', 'Bidder_9']
    Re = ['Bidder_3', 'Bidder_4', 'Bidder_5', 'Bidder_6']
    Lo = ['Bidder_0', 'Bidder_1', 'Bidder_2']

print('SATS Social Welfare:', round(sum_sats, 2))
print('Allocation:')
print(Allocation_sats['good_ids'].to_string())
print('\nValues:')
if model_name == 'MRVM':
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
if model_name == 'MRVM':
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
