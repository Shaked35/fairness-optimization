from random import shuffle

import numpy as np
import pandas as pd

from utils.consts import GENRES


def agg_by_gender_and_genres(input_df: pd.DataFrame, target_genres: str):
    aggregated_df = input_df[input_df[target_genres] == 1].groupby(['user_id', 'gender'])['rating'].agg(
        ['count', 'mean']).reset_index()
    avg_rating_count = aggregated_df.groupby(['gender'])['count'].agg(['mean']).reset_index()
    avg_rating_score = aggregated_df.groupby(['gender'])['mean'].agg(['mean']).reset_index()
    avg_rating_count['type'] = avg_rating_count['gender'].replace({
        'F': 'Ratings per female user',
        'M': 'Ratings per male user'
    })
    avg_rating_score['type'] = avg_rating_score['gender'].replace({
        'F': 'Average rating by men',
        'M': 'Average rating by women'
    })
    avg_rating_count[target_genres] = avg_rating_count['mean']
    avg_rating_score[target_genres] = avg_rating_score['mean']
    results = pd.concat([avg_rating_count[['type', target_genres]], avg_rating_score[['type', target_genres]]])
    results.set_index('type', inplace=True)
    return results


def get_count_by_genres(input_df: pd.DataFrame):
    genre_to_count = {genre: input_df[input_df[genre] == 1]["movie_id"].unique().__len__() for genre in GENRES}
    genre_to_count["type"] = "count"
    results = pd.DataFrame([genre_to_count])
    results.set_index('type', inplace=True)
    return results


def random_k_fold_split(input_df: pd.DataFrame, k: int, test_ratio: float = 0.2):
    unique_users = list(input_df.index)
    number_of_test_users = int(unique_users.__len__() * test_ratio)
    k_folders = []
    for i in range(k):
        shuffle(unique_users)
        test_users = unique_users[:number_of_test_users]
        train_users = unique_users[number_of_test_users:]
        k_folders.append(
            (input_df[input_df.index.isin(train_users)], input_df[input_df.index.isin(test_users)]))
    return k_folders


class MatrixFactorizationWithObservations:
    def __init__(self, num_users, num_items, num_factors=10, lr=0.01, reg=0.01, num_iters=10):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.lr = lr
        self.reg = reg
        self.num_iters = num_iters

    def train(self, train_data, observations):
        # Initialize user and item latent factor matrices
        self.user_factors = np.random.normal(scale=1./self.num_factors, size=(self.num_users, self.num_factors))
        self.item_factors = np.random.normal(scale=1./self.num_factors, size=(self.num_items, self.num_factors))

        # Update the user and item factors iteratively using gradient descent
        for i in range(self.num_iters):
            for user_id, item_id, rating, obs in zip(*train_data, *observations):
                # Compute the prediction error
                error = rating - self.predict(user_id, item_id)

                # Update the user and item factors
                obs_weight = obs + 1  # add 1 to convert observation (0 or 1) to weight (1 or 2)
                self.user_factors[user_id, :] += self.lr * (error * obs_weight * self.item_factors[item_id, :] - self.reg * self.user_factors[user_id, :])
                self.item_factors[item_id, :] += self.lr * (error * obs_weight * self.user_factors[user_id, :] - self.reg * self.item_factors[item_id, :])

    def predict(self, user_id, item_id):
        return np.dot(self.user_factors[user_id, :], self.item_factors[item_id, :])


