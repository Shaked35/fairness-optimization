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
