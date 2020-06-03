#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file implements the class NNMIP (Neural Network Mixed Integer Program). This class is used for solving the deep neural network-based Winner Determination Problem as described in the paper Deep Learning-powered Iterative Combinatorial Auctions
Link: https://arxiv.org/abs/1907.05771 (Section 4).
NN_MIP has the following functionalities:
    0.CONSTRUCTOR: __init__(self, models, L=None)
        models = a dict with (key,value) pairs defined as: ('Bidder_i', an instance of keras.models)
        L = big-M variable which is set gloabally for each node in the DNN-based WDP
    1.METHOD: print_optimal_allocation(self)
        Prints the optimal allocation.
    2.METHOD: solve_mip(self, log_output=False, time_limit=None, mip_relative_gap=None, mip_start=None)
        log_output = boolean for detailed output of cplex when solving the MIP
        time_limit = time limit when solving the MIP
        mip_relative_gap = relative gap limit when solving the MIP
        mip_start = a SolveSolution instance imported from docplex.mp.solution used for a warm start when solving the MIP
        This method returns a docplex.mp.solution.SolveSolution() instance, that can be used as a warm start for the next iteration.
    3.METHOD: __repr__(self)
        Echoe on on your python shell when it evaluates an instances of this class
    4.METHOD: print_mip_constraints(self)
        Prints the MIP constraints of the DNN-based WDP.
    5.METHOD: _get_model_weights(self, key)
        key = the bidder key
        Internal function for getting the estimated parameters of the affine transformations in the neural network for a single bidder.
    6.METHOD:_get_model_layers(self, key, layer_type=None)
        Internal function for getting the layers (keras.layers) of the neural networks.
        layer_type = the layer type: Here, Dense, Input, or Dropout.
    7.METHOD: _clean_weights(self, Wb)
         Wb = return value of _get_model_weights(self, key)
         This method cleans the weights of the affine transformations for a single bidder and sets weights smaller than a threshhold to 0. => MIP are numerically more stable.
    8.METHOD: _add_matrix_constraints(self, i, verbose=False)
         i = the bidder id
         verbose = boolean, level of verbosity when setting up the constraints for the logger.
         This method adds all the matrix constraints for a single bidder, i.e., the recursive contraints as described in the paper in the MIP.
    9.METHOD: initialize_mip(self, verbose=False)
        verbose = boolean, level of verbosity when initializing the MIP for the logger.
        This method initializes the MIP, i.e., dets up all matrix constraints via _add_matrix_constraints(self, i, verbose=False) for all bidders, sets up the feasibility constraints and defines the objective.
    10.METHOD: reset_mip(self):
        This method resets the attribute .Mip of the class.
    11.METHOD: tighten_bounds_IA(self, upper_bound_input, verbose=False)
        upper_bound_input = the upper bound on the input variables, here this is 1 since the input of the neural networks are indicator vectors repesenting the bundle.
        verbose = boolean, level of verbosity when initializing the MIP for the logger.
        This method uses Interval Arithmetic (Box relaxations) to tighten the bounds of the big-M constraints (see Evaluating Robustness of Neural Networks with mixed integer programming. Vincent Tjeng et. al).
    12.METHOD: tighten_bounds_LP(self, upper_bound_input, verbose=False)
        upper_bound_input = the upper bound on the input variables, here this is 1 since the input of the neural networks are indicator vectors repesenting the bundle.
        verbose = boolean, level of verbosity when initializing the MIP for the logger.
        This method uses Interval Arithmetic + LP relaxations to tighten the bounds of the big-M constraints (see Evaluating Robustness of Neural Networks with mixed integer programming. Vincent Tjeng et. al).
    13.METHOD: print_upper_bounds(self, only_zeros=False)
        only_zeros = returns the number of upper bounds equal to zero => this node can be canelled from the MIP formulation.
        This method prints the upper bounds for every node of the neural network.

See example_nn_mip.py for an example of how to use the class NN_MIP.
"""

# Libs
import pandas as pd
import numpy as np
import logging
from collections import OrderedDict
import re
# CPLEX: Here, DOcplex is used for solving the deep neural network-based Winner Determination Problem.
import docplex.mp.model as cpx
# documentation
# http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'
# %% Neural Net Optimization Class


class NNMIP:

    def __init__(self, models, L=None):

        self.M = models[list(models.keys())[0]].get_weights()[0].shape[0]  # number of items in the value model = dimension of input layer
        self.Models = models  # dict of keras models
        # sorted list of bidders
        self.sorted_bidders = list(self.Models.keys())
        self.sorted_bidders.sort()
        self.N = len(models)  # number of bidders
        self.Mip = cpx.Model(name="NeuralNetworksMixedIntegerProgram")  # docplex instance
        self.z = {}  # MIP variable: see paper
        self.s = {}  # MIP variable: see paper
        self.y = {}  # MIP variable: see paper
        self.x_star = np.ones((self.N, self.M))*(-1)  # optimal allocation (-1=not yet solved)
        self.L = L  # global big-M variable: see paper
        self.soltime = None  # timing
        self.z_help = {}  # helper variable for bound tightening
        self.s_help = {}  # helper variable for bound tightening
        self.y_help = {}  # helper variable for bound tightening

        # upper bounds for MIP variables z,s. Lower bounds are 0 in our setting.
        self.upper_bounds_z = OrderedDict(list((bidder_name, [np.array([self.L]*layer.output.shape[1]).reshape(-1, 1) for layer in self._get_model_layers(bidder_name, layer_type=['dense', 'input'])]) for bidder_name in self.sorted_bidders))
        self.upper_bounds_s = OrderedDict(list((bidder_name, [np.array([self.L]*layer.output.shape[1]).reshape(-1, 1) for layer in self._get_model_layers(bidder_name, layer_type=['dense', 'input'])]) for bidder_name in self.sorted_bidders))

    def print_optimal_allocation(self):
        D = pd.DataFrame(self.x_star)
        D.columns = ['Item_{}'.format(j) for j in range(1, self.M+1)]
        D.loc['Sum'] = D.sum(axis=0)
        print(D)

    def solve_mip(self, log_output=False, time_limit=None, mip_relative_gap=None, mip_start=None):
        # add a warm start
        if mip_start is not None:
            self.Mip
            self.Mip.add_mip_start(mip_start)
        # set time limit
        if time_limit is not None:
            self.Mip.set_time_limit(time_limit)
        # set mip relative gap
        if mip_relative_gap is not None:
            self.Mip.parameters.mip.tolerances.mipgap = mip_relative_gap
        logging.debug('Time Limit of %s', self.Mip.get_time_limit())
        logging.debug('Mip relative gap %s', self.Mip.parameters.mip.tolerances.mipgap.get())
        # solve MIP
        Sol = self.Mip.solve(log_output=log_output)
        # get solution details
        self.soltime = Sol.solve_details._time
        logging.debug(self.Mip.get_solve_status())
        logging.debug(self.Mip.get_solve_details())
        logging.debug('Objective Value: %s \n', self.Mip.objective_value)
        # set the optimal allocation
        for i in range(0, self.N):
            for j in range(0, self.M):
                self.x_star[i, j] = self.z[(i, 0, j)].solution_value
        return(Sol)

    def __repr__(self):
        print('################################ OBJECTIVE ################################')
        print(self.Mip.get_objective_expr(), '\n')
        try:
            print('Objective Value: ', self.Mip.objective_value, '\n')
        except Exception:
            print("Not yet solved!\n")
        print('############################# SOLVE STATUS ################################')
        print(self.Mip.get_solve_details())
        print(self.Mip.get_statistics(), '\n')
        try:
            print(self.Mip.get_solve_status(), '\n')
        except AttributeError:
            print("Not yet solved!\n")
        print('########################### OPT ALLOCATION ##############################')
        self.print_optimal_allocation()
        return(' ')

    def print_mip_constraints(self):
        print('############################### CONSTRAINTS ###############################')
        k = 0
        for m in range(0, self.Mip.number_of_constraints):
            if self.Mip.get_constraint_by_index(m) is not None:
                print('({}):   '.format(k), self.Mip.get_constraint_by_index(m))
                k = k+1
            if self.Mip.get_indicator_by_index(m) is not None:
                print('({}):   '.format(k), self.Mip.get_indicator_by_index(m))
                k = k+1
        print('\n')

    def _get_model_weights(self, key):
        Wb = self.Models[key].get_weights()
        return(Wb)

    def _get_model_layers(self, key, layer_type=None):
        Layers = self.Models[key].layers
        if layer_type is not None:
            tmp = [layer.get_config()['name'] for layer in Layers]
            Layers = [Layers[i] for i in [tmp.index(s) for s in tmp if any([x in s for x in layer_type])]]
        return(Layers)

    def _clean_weights(self, Wb):
        for v in range(0, len(Wb)-2, 2):
            Wb[v][abs(Wb[v]) <= 1e-8] = 0
            Wb[v+1][abs(Wb[v+1]) <= 1e-8] = 0
            zero_rows = np.where(np.logical_and((Wb[v] == 0).all(axis=0), Wb[v+1] == 0))[0]
            if len(zero_rows) > 0:
                logging.debug('Clean Weights (rows) %s', zero_rows)
                Wb[v] = np.delete(Wb[v], zero_rows, axis=1)
                Wb[v+1] = np.delete(Wb[v+1], zero_rows)
                Wb[v+2] = np.delete(Wb[v+2], zero_rows, axis=0)
        return(Wb)

    def _add_matrix_constraints(self, i, verbose=False):
        layer = 1
        key = self.sorted_bidders[i]
        Wb = self._clean_weights(self._get_model_weights(key))
        # Wb = self._get_model_weights(key)  # old without weights cleaning
        for v in range(0, len(Wb), 2):  # loop over layers
            if verbose is True:
                logging.debug('\nLayer: %s', layer)
            W = Wb[v].transpose()
            if verbose is True:
                logging.debug('W: %s', W.shape)
            b = Wb[v + 1]
            if verbose is True:
                logging.debug('b: %s', b.shape)
            R, J = W.shape
            # decision variables
            if v == 0:
                self.z.update({(i, 0, j): self.Mip.binary_var(name="x({})_{}".format(i, j)) for j in range(0, J)})  # binary variables for allocation
            self.z.update({(i, layer, r): self.Mip.continuous_var(lb=0, name="z({},{})_{}".format(i, layer, r)) for r in range(0, R)})  # output value variables after activation
            self.s.update({(i, layer, r): self.Mip.continuous_var(lb=0, name="s({},{})_{}".format(i, layer, r)) for r in range(0, R) if (self.upper_bounds_z[key][layer][r][0] != 0 and self.upper_bounds_s[key][layer][r][0] != 0)})  # slack variables
            self.y.update({(i, layer, r): self.Mip.binary_var(name="y({},{})_{}".format(i, layer, r)) for r in range(0, R) if (self.upper_bounds_z[key][layer][r][0] != 0 and self.upper_bounds_s[key][layer][r][0] != 0)})  # binary variables for activation function
            # add constraints
            for r in range(0, R):
                if verbose is True:
                    logging.debug('Row: %s', r)
                    logging.debug('W[r,]: %s', W[r, :])
                    logging.debug('b[r]: %s', b[r])
                    logging.debug('upper z-bound: {}, {}, {}, {}'.format(key, layer, r, self.upper_bounds_z[key][layer][r]))
                    logging.debug('upper s-bound: {}, {}, {}, {}'.format(key, layer, r, self.upper_bounds_s[key][layer][r]))
                if self.upper_bounds_z[key][layer][r][0] == 0:
                    if verbose is True:
                        logging.debug('upper z-bound: {}, {}, {} is equal to zero => add z==0 constraints'.format(key, layer, r))
                    self.Mip.add_constraint(ct=self.z[(i, layer, r)] == 0)
                elif self.upper_bounds_s[key][layer][r][0] == 0:
                    if verbose is True:
                        logging.debug('upper s-bound: {}, {}, {} is equal to zero => add z==Wz_pre + b constraints'.format(key, layer, r))
                    self.Mip.add_constraint(ct=(self.Mip.sum(W[r, j]*self.z[(i, layer-1, j)] for j in range(0, J)) + b[r] == self.z[(i, layer, r)]))
                else:
                    self.Mip.add_constraint(ct=(self.Mip.sum(W[r, j]*self.z[(i, layer-1, j)] for j in range(0, J)) + b[r] == self.z[(i, layer, r)] - self.s[(i, layer, r)]),
                                            ctname="AffineCT_Bidder{}_Layer{}_Row{}".format(i, layer, r))
                    # indicator constraints
                    self.Mip.add_constraint(ct=self.z[(i, layer, r)] <= self.y[(i, layer, r)]*self.upper_bounds_z[key][layer][r][0], ctname="BinaryCT_Bidder{}_Layer{}_Row{}_Z".format(i, layer, r))
                    self.Mip.add_constraint(ct=self.s[(i, layer, r)] <= (1-self.y[(i, layer, r)])*self.upper_bounds_s[key][layer][r][0], ctname="BinaryCT_Bidder{}_Layer{}_Row{}_S".format(i, layer, r))
                if verbose is True:
                    for m in range(0, self.Mip.number_of_constraints):
                        if self.Mip.get_constraint_by_index(m) is not None:
                            logging.debug(self.Mip.get_constraint_by_index(m))
                        if self.Mip.get_indicator_by_index(m) is not None:
                            logging.debug(self.Mip.get_indicator_by_index(m))
            layer = layer + 1

    def initialize_mip(self, verbose=False):
        # pay attention here order is important, thus first sort the keys of bidders!
        # logging.debug(self.sorted_bidders)
        # linear matrix constraints: Wz^(i-1)+b = z^(i)-s^(i)
        for i in range(0, self.N):
            self._add_matrix_constraints(i, verbose=verbose)
        # allocation constraints for x^i's
        for j in range(0, self.M):
            self.Mip.add_constraint(ct=(self.Mip.sum(self.z[(i, 0, j)] for i in range(0, self.N)) <= 1), ctname="FeasabilityCT_x({})".format(j))
        # add objective
        objective = self.Mip.sum(self.z[(i, (len(self._get_model_layers(self.sorted_bidders[i], layer_type=['dense']))), 0)] for i in range(0, self.N))
        self.Mip.maximize(objective)
        self.Mip.parameters.mip.tolerances.integrality.set(1e-8)
        logging.debug('Mip initialized')

    def reset_mip(self):
        self.Mip = cpx.Model(name="MIP")

    def tighten_bounds_IA(self, upper_bound_input, verbose=False):
        for bidder in self.sorted_bidders:
            logging.debug('Tighten bounds with IA for %s', bidder)
            Wb_total = self._clean_weights(self._get_model_weights(bidder))
            k = 0
            for j in range(len(self._get_model_layers(bidder, layer_type=['dense', 'input']))):  # loop over layers including input layer
                if j == 0:
                    self.upper_bounds_z[bidder][j] = np.array(upper_bound_input).reshape(-1, 1)
                    self.upper_bounds_s[bidder][j] = np.array(upper_bound_input).reshape(-1, 1)
                else:
                    W_plus = np.maximum(Wb_total[k].transpose(), 0)
                    W_minus = np.minimum(Wb_total[k].transpose(), 0)
                    self.upper_bounds_z[bidder][j] = np.ceil(np.maximum(W_plus @ self.upper_bounds_z[bidder][j-1] + Wb_total[k+1].reshape(-1, 1), 0)).astype(int)   # upper bound for z
                    self.upper_bounds_s[bidder][j] = np.ceil(np.maximum(-(W_minus @ self.upper_bounds_z[bidder][j-1] + Wb_total[k+1].reshape(-1, 1)), 0)).astype(int)  # upper bound  for s
                    k = k+2
        if verbose is True:
            logging.debug('Upper Bounds z:')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_z.items()}).fillna('-').items():
                logging.debug(v)
            logging.debug('\nUpper Bounds s:')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_s.items()}).fillna('-').items():
                logging.debug(v)

    def tighten_bounds_LP(self, upper_bound_input, verbose=False):

        for bidder in self.sorted_bidders:
            logging.debug('Tighten bounds with LPs for %s', bidder)
            i = int(re.findall(r'\d+', bidder)[0])
            Wb_total = self._clean_weights(self._get_model_weights(bidder))
            for layer in range(len(self._get_model_layers(bidder, layer_type=['dense', 'input']))):  # loop over layers including input layer
                if layer == 0:  # input layer bounds given
                    self.upper_bounds_z[bidder][layer] = np.array(upper_bound_input).reshape(-1, 1)
                    self.upper_bounds_s[bidder][layer] = np.array(upper_bound_input).reshape(-1, 1)
                elif layer == 1:   # first hidden layer no LP needed, can be done like IA
                    W_plus = np.maximum(Wb_total[0].transpose(), 0)
                    W_minus = np.minimum(Wb_total[0].transpose(), 0)
                    self.upper_bounds_z[bidder][layer] = np.ceil(np.maximum(W_plus.sum(axis=1).reshape(-1, 1) + Wb_total[1].reshape(-1, 1), 0)).astype(int)   # upper bound for z
                    self.upper_bounds_s[bidder][layer] = np.ceil(np.maximum(-(W_minus.sum(axis=1).reshape(-1, 1) + Wb_total[1].reshape(-1, 1)), 0)).astype(int)  # upper bound  for s
                else:
                    for k in range(0, len(self.upper_bounds_z[bidder][layer])):
                        if (self.upper_bounds_z[bidder][layer][k][0] == 0 and self.upper_bounds_s[bidder][layer][k][0] == 0):
                            continue
                        helper_Mip = cpx.Model(name="LPBounds")
                        pre_layer = 1
                        for v in range(0, 2*(layer-1), 2):  # loop over prelayers before layer
                            W = Wb_total[v].transpose()
                            b = Wb_total[v + 1]
                            ROWS, COLUMNS = W.shape
                            # initialize decision variables
                            if v == 0:
                                self.z_help.update({(i, 0, j): helper_Mip.binary_var(name="x({})_{}".format(i, j)) for j in range(0, COLUMNS)})  # binary variables for allocation
                            self.z_help.update({(i, pre_layer, r): helper_Mip.continuous_var(lb=0, name="z({},{})_{}".format(i, pre_layer, r)) for r in range(0, ROWS)})  # output value variables after activation
                            self.s_help.update({(i, pre_layer, r): helper_Mip.continuous_var(lb=0, name="s({},{})_{}".format(i, pre_layer, r)) for r in range(0, ROWS) if (self.upper_bounds_z[bidder][pre_layer][r][0] != 0 and self.upper_bounds_s[bidder][pre_layer][r][0] != 0)})  # slack variables
                            self.y_help.update({(i, pre_layer, r): helper_Mip.continuous_var(lb=0, ub=1, name="y({},{})_{}".format(i, pre_layer, r)) for r in range(0, ROWS) if (self.upper_bounds_z[bidder][pre_layer][r][0] != 0 and self.upper_bounds_s[bidder][pre_layer][r][0] != 0)})  # relaxed binary variables for activation function
                            # add constraints
                            for r in range(0, ROWS):
                                if self.upper_bounds_z[bidder][pre_layer][r][0] == 0:
                                    if verbose is True:
                                        logging.debug('upper z-bound: {}, {}, {} is equal to zero => add z==0 constraints'.format(bidder, pre_layer, r))
                                    helper_Mip.add_constraint(ct=self.z_help[(i, pre_layer, r)] == 0)
                                elif self.upper_bounds_s[bidder][pre_layer][r][0] == 0:
                                    if verbose is True:
                                        logging.debug('upper s-bound: {}, {}, {} is equal to zero => add z==Wz_pre + b constraints'.format(bidder, pre_layer, r))
                                    helper_Mip.add_constraint(ct=(helper_Mip.sum(W[r, j]*self.z_help[(i, pre_layer-1, j)] for j in range(0, COLUMNS)) + b[r] == self.z_help[(i, pre_layer, r)]))
                                else:
                                    helper_Mip.add_constraint(ct=(helper_Mip.sum(W[r, j]*self.z_help[(i, pre_layer-1, j)] for j in range(0, COLUMNS)) + b[r] == self.z_help[(i, pre_layer, r)] - self.s_help[(i, pre_layer, r)]),
                                                              ctname="AffineCT_Bidder{}_Layer{}_Row{}".format(i, pre_layer, r))
                                    # relaxed indicator constraints
                                    helper_Mip.add_constraint(ct=self.z_help[(i, pre_layer, r)] <= self.y_help[(i, pre_layer, r)]*self.upper_bounds_z[bidder][pre_layer][r][0], ctname="RelaxedBinaryCT_Bidder{}_Layer{}_Row{}_Z".format(i, pre_layer, r))
                                    helper_Mip.add_constraint(ct=self.s_help[(i, pre_layer, r)] <= (1-self.y_help[(i, pre_layer, r)])*self.upper_bounds_s[bidder][pre_layer][r][0], ctname="RelaxedBinaryCT_Bidder{}_Layer{}_Row{}_S".format(i, pre_layer, r))
                            pre_layer = pre_layer + 1

                        # final extra row constraint
                        W = Wb_total[2*(layer-1)].transpose()
                        b = Wb_total[2*(layer-1) + 1]
                        self.z_help.update({(i, layer, k): helper_Mip.continuous_var(lb=0, name="z({},{})_{}".format(i, layer, k))})
                        self.s_help.update({(i, layer, k): helper_Mip.continuous_var(lb=0, name="s({},{})_{}".format(i, layer, k))})
                        if self.upper_bounds_z[bidder][layer][k][0] == 0:
                            helper_Mip.add_constraint(ct=(helper_Mip.sum(W[k, j]*self.z_help[(i, layer-1, j)] for j in range(0, W.shape[1])) + b[k] == -self.s_help[(i, layer, k)]))
                        elif self.upper_bounds_s[bidder][layer][k][0] == 0:
                            helper_Mip.add_constraint(ct=(helper_Mip.sum(W[k, j]*self.z_help[(i, layer-1, j)] for j in range(0, W.shape[1])) + b[k] == self.z_help[(i, layer, k)]))
                        else:
                            self.y_help.update({(i, layer, k): helper_Mip.continuous_var(lb=0, ub=1, name="y({},{})_{}".format(i, layer, k))})  # relaxed binary variable for activation function for final row constraint
                            helper_Mip.add_constraint(ct=(helper_Mip.sum(W[k, j]*self.z_help[(i, layer-1, j)] for j in range(0, W.shape[1])) + b[k] == self.z_help[(i, layer, k)] - self.s_help[(i, layer, k)]),
                                                      ctname="FinalAffineCT_Bidder{}_Layer{}_Row{}".format(i, layer, k))
                            # final relaxed indicator constraints
                            helper_Mip.add_constraint(ct=self.z_help[(i, layer, k)] <= self.y_help[(i, layer, k)]*self.upper_bounds_z[bidder][layer][k][0], ctname="FinalRelaxedBinaryCT_Bidder{}_Layer{}_Row{}_Z".format(i, layer, k))
                            helper_Mip.add_constraint(ct=self.s_help[(i, layer, k)] <= (1-self.y_help[(i, layer, k)])*self.upper_bounds_s[bidder][layer][k][0], ctname="FinalRelaxedBinaryCT_Bidder{}_Layer{}_Row{}_S".format(i, layer, k))
                        # add objective for z bound only if current bounds larger than zero
                        helper_Mip.parameters.mip.tolerances.integrality.set(1e-8)
                        if self.upper_bounds_z[bidder][layer][k][0] != 0:
                            helper_Mip.maximize(self.z_help[(i, layer, k)])
                            helper_Mip.solve()
                            self.upper_bounds_z[bidder][layer][k][0] = np.ceil(self.z_help[(i, layer, k)].solution_value).astype(int)
                        if self.upper_bounds_s[bidder][layer][k][0] != 0:
                            # add objective for s bound only if current bounds larger than zero
                            helper_Mip.maximize(self.s_help[(i, layer, k)])
                            helper_Mip.solve()
                            self.upper_bounds_s[bidder][layer][k][0] = np.ceil(self.s_help[(i, layer, k)].solution_value).astype(int)
                        del helper_Mip
        if verbose is True:
            logging.debug('Upper Bounds z:')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_z.items()}).fillna('-').items():
                logging.debug(v)
            logging.debug('\nUpper Bounds s:')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_s.items()}).fillna('-').items():
                logging.debug(v)

    def print_upper_bounds(self, only_zeros=False):
        zeros = 0
        for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_z.items()}).fillna('-').items():
            if not only_zeros:
                print('Upper Bounds z:')
                print(v)
                print()
            zeros = zeros + sum([np.sum(x == 0) for x in v])
        for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_s.items()}).fillna('-').items():
            if not only_zeros:
                print('Upper Bounds s:')
                print(v)
                print()
            zeros = zeros + sum([np.sum(x == 0) for x in v])
        print('Number of Upper bounds equal to 0: ', zeros)


print('NN_MIP Class imported')
