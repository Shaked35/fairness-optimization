from datetime import time

import numpy as np

from fairness_methods.methods import FairnessMethods
from utils.metrics import RecommendationSystemMetrics


class FairnessMF(object):

    def __init__(self, df_train, disadvantaged_group, advantaged_group, number_of_items, n_factors=10, only_bias=False):
        self.df_train = df_train
        self.n_factors = n_factors
        self.only_bias = only_bias
        self.n_rows, self.n_cols = df_train.shape
        self.disadvantaged_group = disadvantaged_group
        self.advantaged_group = advantaged_group
        self.number_of_items = number_of_items

        self.original_bias_global = np.sum(df_train) / np.count_nonzero(df_train)
        self.original_bias_rows = np.sum(df_train, axis=1) / np.count_nonzero(df_train, axis=1)
        self.original_bias_cols = np.sum(df_train, axis=0) / np.count_nonzero(df_train, axis=0)

        # "reset" initialization
        self.initialize_params()

    def initialize_params(self):
        if self.only_bias:
            self.ui = np.zeros((self.n_rows, self.n_factors))  # the u and v matix are zeros if only bias
            self.vj = np.zeros((self.n_cols, self.n_factors))
        else:  # otherwise - start with radom vals:
            self.ui = np.random.normal(scale=1. / self.n_factors, size=(self.n_rows, self.n_factors))
            self.vj = np.random.normal(scale=1. / self.n_factors, size=(self.n_cols, self.n_factors))

        # initilize bias
        self.bias_global = self.original_bias_global
        self.bias_rows = np.random.rand(self.n_rows)  # randon as rows len
        self.bias_cols = np.random.rand(self.n_cols)  # random as cols len

    def fit(self,
            n_iterations=1,
            learning_rate=1e-1,
            regularization=1e-2,
            convergence=1e-5,
            error='RMSE',
            initilize_training=True,
            verbose=True):

        self.n_iterations = n_iterations
        self.α = learning_rate
        self.λ = regularization
        self.ϵ = convergence
        self.error = error

        if initilize_training:
            self.initialize_params()

        self.history = []

        start_time = time.time()

        for current_iteration in range(self.n_iterations):

            self.history.append(self.get_rmse(self.df_train))

            # printing
            if verbose:
                print('iteration: ', current_iteration, ' total error:\n', self.history[-1])
            # convergence
            if current_iteration != 0 and self.converging():

                if verbose:
                    print('converged...')
                break
            self.optim_GD()

        self.fit_time = time.time() - start_time

    def converging(self):
        return np.abs(
            self.history[-1] - self.history[-2]) < self.ϵ  # if error wont decreasing by ϵ factor -> stop training

    def optim_GD(self):
        eij = self.get_fairness_eij()  # calc error
        if not self.only_bias:  # if not only bias, traing GD with regularization

            ui_copy = self.ui.copy()  # for simultaneous update
            self.ui += self.α * (np.dot(eij,self.vj) - self.λ * self.ui)
            self.vj += self.α * (np.dot(eij.T, ui_copy) - self.λ * self.vj)

        # update biases:
        self.bias_rows += self.α * (eij.mean(axis=1) - (self.λ * self.bias_rows))
        self.bias_cols += self.α * (eij.mean(axis=0) - (self.λ * self.bias_cols))

    def get_fairness_eij(self):
        # add biases
        pred = self.predict()
       # calc error (delta between label and pred)
        fairness = FairnessMethods().calculate_val_score(pred, self.df_train, self.disadvantaged_group, self.advantaged_group, self.number_of_items)
        return self.df_train - pred + fairness

    def predict(self):
        # add biases
        pred = np.dot(self.ui, self.vj.T)
        pred += self.bias_global
        pred += self.bias_rows[:, np.newaxis]
        pred += self.bias_cols[np.newaxis, :]
        # return pred
        return pred

    def get_rmse(self, true):
        preds = self.predict()
        return RecommendationSystemMetrics().RMSE(true_array=true, pred_array=preds)