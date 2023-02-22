from random import shuffle

import numpy as np
import pandas as pd

from utils.consts import GENRES


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


def get_one_hot_encoding_genres(genres: list):
    """
    :param genres: list of genres names in row
    :return: one hot encoded genres
    """
    return [1 if g in genres else 0 for g in GENRES]

