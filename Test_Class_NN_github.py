#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file presents examples of how to use the NN class from file Class_NN_github.py
"""

# Libs
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler

# Own Modules
from PySats import PySats
from Class_NN_github import NN
import util_github as util


__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'

# %%
model_name = 'LSVM'  # select a model name from ('LSVM', 'GSVM', 'MRVM')
value_model = PySats.getInstance().create_lsvm(seed=1)  # create PySats instance
bidder_ids = list(value_model.get_bidder_ids())
scaler = None  # set scaler
# scaler = MinMaxScaler()  # set scaler
n_train = 1000  # number of training bundle-value pairs
n_valid = 100  # number of test bundle-value pairs
# Generate training set of bundle-value pairs sampled uniformly at random for each bidder D (here no test sets). For each (X_train,Y_train) set the null bundle is automatically added.
# The structure of D is as follows:  D = [OrderedDict((Bidder_1:[X^1_train,Y^1_train]),...,(Bidder_n:[X^n_train,Y^n_train])],scaler]
D = util.data_preparation_UNIF(model_name=model_name, value_model=value_model, n_train=n_train, n_valid=n_valid, bidder_ids=bidder_ids, scaler=scaler)
epochs = 512  # epochs for training the nerual network
batch_size = 30  # batch size for training the neural network
regularization_type = 'l1_l2'  # reularization parameter of the affine mappings between the layers
# %%
dropout = True  # dropout activated
dropout_rate = 0.1  # droput rate
# define model parameters, e.g.,(r=1e-05, lr=0.01, architecture=[10, 10, 10], dropout, dropout_rate), this defines a three hidden layer neural network with 10 hidden nodes each.
NN_parameters = OrderedDict([('Bidder_0', (1e-05, 0.01, [10, 10, 10], dropout, dropout_rate)), ('Bidder_1', (1e-05, 0.01, [32, 32], dropout, dropout_rate)),
                             ('Bidder_2', (1e-05, 0.01, [32, 32], dropout, dropout_rate)), ('Bidder_3', (1e-05, 0.01, [32, 32], dropout, dropout_rate)),
                             ('Bidder_4', (1e-05, 0.01, [32, 32], dropout, dropout_rate)), ('Bidder_5', (1e-05, 0.01, [32, 32], dropout, dropout_rate))])
# %%  Test the class for a single bidder.
key = 'Bidder_0'  # set bidder
Bids = D[0]
fitted_scaler = D[1]
value = Bids[key]  # take the training set (X_train,Y_train) from bidder 0 which is stored in D[0]['Bidder_0'].

# create instance from class NN for a single bidder specyfied by key.
neural_net = NN(model_parameters=NN_parameters[key], X_train=value[0], Y_train=value[1], scaler=fitted_scaler)
# initialize model
neural_net.initialize_model(regularization_type=regularization_type)
# fit model and store losses
if n_valid != 0:
    loss = neural_net.fit(epochs=epochs, batch_size=batch_size, X_valid=value[2], Y_valid=value[3])
else:
    loss = neural_net.fit(epochs=epochs, batch_size=batch_size, X_valid=None, Y_valid=None)
loss  # loss info (tr, val, tr_orig, val_orig)
neural_net.history.history['loss']  # loss evolution over epochs
neural_net.loss_info(batch_size=n_train, plot=True, scale='log')  # loss info with plots
