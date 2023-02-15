import time

import numpy as np

from utils.metrics import RecommendationSystemMetrics


class BasicMF(object):

    def __init__(self, df_train, n_factors=10, only_bias=False):
        self.df_train = df_train
        self.n_factors = n_factors
        self.only_bias = only_bias
        self.n_rows, self.n_cols = df_train.shape

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
        for row, col in [(i, j) for i in range(self.n_rows) for j in range(self.n_cols) if
                         self.df_train[i, j] > 0]:  # loop over matrix
            eij = self.get_eij(row, col)  # calc error
            if not self.only_bias:  # if not only bias, traing GD with regularization

                ui_copy = self.ui.copy()  # for simultaneous update
                self.ui[row, :] += self.α * (eij * self.vj[col, :] - self.λ * self.ui[row, :])
                self.vj[col, :] += self.α * (eij * ui_copy[row, :] - self.λ * self.vj[col, :])

            # update biases:
            self.bias_rows[row] += self.α * (eij - self.λ * self.bias_rows[row])
            self.bias_cols[col] += self.α * (eij - self.λ * self.bias_cols[col])

    def get_eij(self, row, col):
        # add biases
        pred = np.dot(self.ui[row], self.vj[col].T)
        pred += self.bias_global
        pred += self.bias_rows[row, np.newaxis]
        pred += self.bias_cols[np.newaxis, col]
        # calc error (delta between label and pred)
        return self.df_train[row, col] - pred

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
