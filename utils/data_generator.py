import os.path
import shutil
import urllib.request

import numpy as np
import pandas as pd
from tabulate import tabulate

from utils.consts import *
from utils.util import get_one_hot_encoding_genres


def generate_synthetic_data(observation_model, user_distribution, num_users=NUM_USERS, num_items=NUM_ITEMS):
    """
    This function generates synthetic data for the given user and item distribution.
    :param observation_model: uniform or unbalanced observation data
    :param user_distribution: imbalanced or biased user distribution
    :param num_users: number of users (rows)
    :param num_items: number of items (columns)
    :return:
    ratings_df -> which contains columns of user_id, item_id and rating
    item_id_to_group -> map of item id to item type ('Fem', 'STEM', 'Masc')
    """
    # Define the block-model parameters for rating probability
    L = np.array([[0.8, 0.2, 0.2],
                  [0.8, 0.8, 0.2],
                  [0.2, 0.8, 0.8],
                  [0.2, 0.2, 0.8]])
    rating_probability = tabulate(pd.DataFrame(L, columns=ITEM_GROUPS, index=USER_GROUPS),
                                  ITEM_GROUPS, tablefmt="fancy_grid", showindex=True)

    # Define the block-model parameters for observation probability
    # observation_model is 'unbalanced'
    O = np.array([[0.6, 0.2, 0.1],
                  [0.3, 0.4, 0.2],
                  [0.1, 0.3, 0.5],
                  [0.05, 0.5, 0.35]])
    if observation_model == 'uniform':
        O = np.full((len(USER_GROUPS), len(ITEM_GROUPS)), 0.4)

    observation_probability = tabulate(pd.DataFrame(O, columns=ITEM_GROUPS, index=USER_GROUPS),
                                       ITEM_GROUPS, tablefmt="fancy_grid", showindex=True)
    print("****************************rating probability****************************\n")
    print(rating_probability)
    print("****************************observation probability****************************\n")
    print(observation_probability)
    # Define the user group distribution
    # user_distribution is 'biased'
    num_users_per_group = int(num_users / len(USER_GROUPS))
    user_group_dist = [num_users_per_group] * len(USER_GROUPS)
    if user_distribution == 'imbalanced':
        # where 0.4 of the population is in W, 0.1 in WS, 0.4 in MS, and 0.1 in M
        user_group_dist = [int(0.4 * num_users), int(0.1 * num_users), int(0.4 * num_users), int(0.1 * num_users)]

    # Generate user and item ids
    user_ids = np.repeat(np.arange(len(USER_GROUPS)), user_group_dist)
    item_ids = np.tile(np.arange(len(ITEM_GROUPS)), int(num_items / len(ITEM_GROUPS)))

    # Shuffle the user and item ids
    np.random.shuffle(user_ids)
    np.random.shuffle(item_ids)

    # Generate preference and observation data based on the block models
    preference = np.zeros((num_users, num_items))
    observations = np.zeros((num_users, num_items))
    user_gender_rows = []
    for i in range(num_users):
        user_type = user_ids[i]
        user_gender = "M" if user_type < 2 else "F"
        user_gender_rows.append({"user_id": i, "gender": user_gender})
        for j in range(num_items):
            prob_rating = L[user_type, item_ids[j]]
            rating = np.random.choice([-1, 1], p=[1 - prob_rating, prob_rating])
            preference[i, j] = rating

            prob_observation = O[user_type, item_ids[j]]
            observation = np.random.choice([0, 1], p=[1 - prob_observation, prob_observation])
            observations[i, j] = observation

    # combine preference with observations and set scores
    ratings = preference.copy()
    ratings = 2 * ratings - 1
    ratings[observations == 0] = 0

    # convert np matrix to pandas dataframe with 3 column 'user_id', 'item_id', 'rating'
    users = np.repeat(np.arange(ratings.shape[0]), len(ratings.flatten()) / len(np.arange(ratings.shape[0])))
    items = np.tile(np.arange(ratings.shape[1]), int(len(ratings.flatten()) / len(np.arange(ratings.shape[1]))))
    ratings_scores = ratings.flatten()
    ratings_df = pd.DataFrame()
    ratings_df["rating"] = ratings_scores
    ratings_df["user_id"] = users.astype(int)
    ratings_df["item_id"] = items.astype(int)
    user_gender_df = pd.DataFrame(user_gender_rows)
    user_gender_df["user_id"] = user_gender_df["user_id"].astype(int)
    ratings_df = ratings_df.merge(user_gender_df, on=["user_id"], how="left")
    item_id_to_group = {index: group for index, group in enumerate(item_ids)}
    return ratings_df, item_id_to_group



def generate_real_data():
    """
    This method generates new dataset from https://files.grouplens.org/datasets/movielens/ml-1m.zip that
    contains users ratings on different movies
    :return: ratings dataframe
    """
    if not os.path.isfile(ZIP_NAME):
        urllib.request.urlretrieve(ML_URL, ZIP_NAME)
        shutil.unpack_archive(ZIP_NAME, RESOURCE_DIR)
    # load movie rating data
    ratings = pd.read_csv(os.path.join(LOCAL_DATASET_LOCATION, 'ratings.dat'), sep='::',
                          header=None, names=RATING_COLUMNS, encoding='ISO-8859-1', engine='python')
    movies = pd.read_csv(os.path.join(LOCAL_DATASET_LOCATION, 'movies.dat'), sep='::',
                         header=None, engine='python', names=['movie_id', 'title', 'genres'], encoding='ISO-8859-1')
    users = pd.read_csv(os.path.join(LOCAL_DATASET_LOCATION, 'users.dat'), sep='::',
                        header=None, engine='python', names=['user_id', 'gender', 'age', 'occupation', 'zip'],
                        encoding='ISO-8859-1')
    user_rating_counts = ratings.groupby("user_id").count()
    # filter users without enough ratings
    users_with_enough_ratings = list(user_rating_counts[user_rating_counts["movie_id"] >= MIN_USER_RATED].index)
    ratings = ratings[ratings["user_id"].isin(users_with_enough_ratings)]
    genres_matrix = movies["genres"].apply(lambda genres: get_one_hot_encoding_genres(genres))
    movies = pd.concat([movies, pd.DataFrame(list(genres_matrix), columns=GENRES)], axis=1)
    # filter movies from genres list
    movies["relevant_movies"] = list(movies[GENRES].sum(axis=1))
    movies = movies[movies["relevant_movies"] >= 1]
    ratings = ratings[ratings["movie_id"].isin(movies["movie_id"].values)]
    return ratings.merge(movies[["movie_id"] + GENRES], on=["movie_id"], how="left").merge(users, on=["user_id"],
                                                                                           how="left")
