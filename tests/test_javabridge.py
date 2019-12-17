#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
This file shows examples of how to use the jave-python bridge file pysats.py and create auction instances of the three valuation models Local Synergy Value Model (LSVM), Global Synergy Value Model (GSVM), Multi Region Value Model (MRVM)
"""

# Libs
import numpy as np

# Own modules
from source.pysats import PySats

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'

################################################
# LSVM, GSVM and MRVM class & functionalities: #
################################################
# %%
# create instance with the default parmetrization of bidder types.
G = PySats.getInstance().create_gsvm(seed=1, number_of_national_bidders=1, number_of_regional_bidders=5)
L = PySats.getInstance().create_lsvm(seed=1, number_of_national_bidders=1, number_of_regional_bidders=5)
M = PySats.getInstance().create_mrvm(seed=1, number_of_national_bidders=3, number_of_regional_bidders=4, number_of_local_bidders=3)
# %%
# get bidder ids. Return: dict.keys
print('GSVM')
print(G.get_bidder_ids())
print('LSVM')
print(L.get_bidder_ids())
print('MRVM')
print(M.get_bidder_ids())
# %%
# get good ids. Return: dict.keys
print('GSVM')
print(G.get_good_ids())
print('LSVM')
print(L.get_good_ids())
print('MRVM')
print(M.get_good_ids())
# %%
# Query some value. Return: float
bidder_id = 2
print(bidder_id)
print('GSVM')
bundle = np.random.choice(2, 18)
print('Bundle:', bundle)
value = G.calculate_value(bidder_id, bundle)
print('Value:', value)
# Query some value. return float
print('LSVM')
bundle = np.random.choice(2, 18)
print('Bundle:', bundle)
value = L.calculate_value(bidder_id, bundle)
print('Value:', value)
# Query some value. return float
print('MRVM')
bundle = np.random.choice(2, 98)
print('Bundle:', bundle)
value = M.calculate_value(bidder_id, bundle)
print('Value:', value)
# %%
# Generate some bids. Return: 2 dimesional list
bids1 = G.get_random_bids(bidder_id=0, number_of_bids=5)
print(bids1)
print(np.array(bids1))
bids2 = L.get_random_bids(bidder_id=1, number_of_bids=5)
print(bids2)
print(np.array(bids2))
bids3 = M.get_random_bids(bidder_id=8, number_of_bids=5)
print(bids3)
print(np.array(bids3))

# %%
# get efficient allocation and efficiency. Return: dict
allocation, social_welfare = G.get_efficient_allocation()
print('Allocation:', allocation, '\n')
print('Social Welfare:', social_welfare, '\n')
allocation, social_welfare = L.get_efficient_allocation()
print('Allocation:', allocation, '\n')
print('Social Welfare:', social_welfare, '\n')
allocation, social_welfare = M.get_efficient_allocation()
print('Allocation:', allocation, '\n')
print('Social Welfare:', social_welfare, '\n')
