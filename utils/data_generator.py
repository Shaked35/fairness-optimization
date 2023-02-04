import os.path
import shutil
import urllib.request

import pandas as pd

import numpy as np

from utils.consts import *


def generate_synthetic_data(L, O, num_users, num_items):
    r = np.zeros((num_users, num_items))
    for i in range(num_users):
        for j in range(num_items):
            r_ij = np.random.binomial(1, L[i % L.shape[0], j % L.shape[1]])
            if np.random.binomial(1, O[i % O.shape[0], j % O.shape[1]]) == 1:
                r[i, j] = r_ij
            else:
                r[i, j] = -1
    return r


def get_one_hot_encoding_genres(genres):
    return [1 if g in genres else 0 for g in GENRES]


def generate_real_data():
    if not os.path.isfile("../../resource/ml-1m.zip"):
        urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-1m.zip",
                                   "../../resource/ml-1m.zip")
        shutil.unpack_archive("../../resource/ml-1m.zip", "../../resource/ml-1m")
    # load movie rating data
    ratings = pd.read_table('../../resource/ml-1m/ml-1m/ratings.dat', sep='::', header=None, names=RATING_COLUMNS,
                            encoding='ISO-8859-1')
    movies = pd.read_table('../../resource/ml-1m/ml-1m/movies.dat', sep='::', header=None,
                           names=['movie_id', 'title', 'genres'], encoding='ISO-8859-1')
    users = pd.read_table('../../resource/ml-1m/ml-1m/users.dat', sep='::', header=None,
                          names=['user_id', 'gender', 'age', 'occupation', 'zip'], encoding='ISO-8859-1')
    user_rating_counts = ratings.groupby("user_id").count()
    # filter users without enough ratings
    users_with_enough_ratings = list(user_rating_counts[user_rating_counts["movie_id"] >= MIN_USER_RATED].index)
    ratings = ratings[ratings["user_id"].isin(users_with_enough_ratings)]
    genres_matrix = movies["genres"].apply(lambda genres: get_one_hot_encoding_genres(genres))
    movies = pd.concat([movies, pd.DataFrame(list(genres_matrix), columns=GENRES)], axis=1)
    return ratings.merge(movies[["movie_id"] + GENRES], on=["movie_id"], how="left").merge(users, on=["user_id"],
                                                                                           how="left")
