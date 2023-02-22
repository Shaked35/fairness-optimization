import numpy as np
from scipy.sparse import coo_matrix


class BasicMatrixFactorization:
    def __init__(self, n_users, n_items, n_user_groups=2, n_items_groups=5, n_factors=20, lambda_=0.001, alpha=0.001,
                 epochs=10):
        self.group_items = None
        self.group_users = None
        self.q = None
        self.p = None
        self.bi = None
        self.bu = None
        self.mean_rating = None
        self.group_item = None
        self.group_user = None

        self.n_users = n_users
        self.n_items = n_items
        self.n_user_groups = n_user_groups
        self.n_items_groups = n_items_groups
        self.n_factors = n_factors
        self.lambda_ = lambda_
        self.alpha = alpha
        self.n_epochs = epochs
        print(f"init model with n_users={n_users}, n_items={n_items}, n_user_groups={n_user_groups}")

    def fit(self, train_data, group_user, group_item):
        self.group_user = group_user
        self.group_item = group_item
        self.mean_rating = train_data.mean()
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        self.p = np.random.normal(scale=1. / self.n_factors, size=(self.n_users, self.n_factors))
        self.q = np.random.normal(scale=1. / self.n_factors, size=(self.n_items, self.n_factors))
        self.group_users = np.random.normal(scale=1. / self.n_factors, size=(self.n_user_groups, self.n_factors))
        self.group_items = np.random.normal(scale=1. / self.n_factors, size=(self.n_items_groups, self.n_factors))
        matrix = coo_matrix(train_data)
        for epoch in range(self.n_epochs):
            for u, i, r in zip(matrix.row, matrix.col, matrix.data):
                group_u = self.group_user[u]
                group_i = self.group_item[i]
                err = r - self.predict(u, i, group_u, group_i)
                self.bu[u] += self.alpha * (err - self.lambda_ * self.bu[u])
                self.bi[i] += self.alpha * (err - self.lambda_ * self.bi[i])
                self.p[u, :] += self.update_user_item(err, i, u, update_user=True)
                self.q[i, :] += self.update_user_item(err, i, u, update_user=False)
                self.group_users[group_u, :] += self.alpha * (
                        err * self.group_items[group_i, :] - self.lambda_ * self.group_users[group_u, :])
                self.group_items[group_i, :] += self.alpha * (
                        err * self.group_users[group_u, :] - self.lambda_ * self.group_items[group_i, :])

    def update_user_item(self, err, i, u, update_user):
        if update_user:
            return self.alpha * (err * self.q[i, :] - self.lambda_ * self.p[u, :])
        else:
            return self.alpha * (err * self.p[u, :] - self.lambda_ * self.q[i, :])

    def predict(self, u, i, group_u, group_i):
        pred = self.mean_rating + self.bu[u] + self.bi[i] + self.group_users[group_u, :].dot(
            self.group_items[group_i, :].T)
        pred += self.p[u, :].dot(self.q[i, :].T)
        return pred

    def predictions(self):
        predictions = np.zeros((self.n_users, self.n_items))
        for u in range(self.n_users):
            for i in range(self.n_items):
                group_u = self.group_user[u]
                group_i = self.group_item[i]
                pred = self.mean_rating + self.bu[u] + self.bi[i] + self.group_users[group_u, :].dot(
                    self.group_items[group_i, :].T)
                pred += self.p[u, :].dot(self.q[i, :].T)
                predictions[u][i] = pred
        return predictions
