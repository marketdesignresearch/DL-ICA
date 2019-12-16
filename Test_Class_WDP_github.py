#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file presents examples of how to use the WDP class from file Class_WDP_github.py
"""

# Libs
import numpy as np

# Own Modules
from Class_WDP_github import WDP

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'

# %% Toy example with artifical bids

# generate arbitrary bids
bids = []
N = 3  # 3 bidders
M = 4  # 3 items
K = [3, 4, 5]  # number of elicited bids per bidder
# generate random bundle-value pairs
for i in range(0, N):
    bids.append(np.concatenate((np.random.randint(2, size=(K[i], M)), (1000*np.random.random_sample(K[i])).reshape(-1, 1)), axis=1))
    print('Elicited Bids from Bidder {}:'.format(i))
    print(bids[i])
    print()

X = WDP(bids)  # initialize WDP class
X.initialize_mip(verbose=True)   # initialize MIP
X  # not yet solved
X.solve_mip()   # solve MIP
X
