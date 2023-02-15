import numpy as np
from numpy import mean


class FairnessMethods(object):

    @staticmethod
    def calculate_average_values(advantaged_group, disadvantaged_group, number_of_items, predictions, ratings, func):
        values = []
        for item_pos in range(number_of_items):
            e_g_y_j = np.nanmean(predictions[disadvantaged_group][:, item_pos])
            e_neg_g_y_j = np.nanmean(predictions[advantaged_group][:, item_pos])
            e_g_r_j = np.nanmean(ratings[disadvantaged_group][:, item_pos])
            e_neg_g_r_j = np.nanmean(ratings[advantaged_group][:, item_pos])
            values.append(func(e_g_y_j, e_g_r_j, e_neg_g_y_j, e_neg_g_r_j))
        return np.nanmean(values)

    @staticmethod
    def calculate_val_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                            advantaged_group: np.array, number_of_items: int):
        def func(e_g_y_j, e_g_r_j, e_neg_g_y_j, e_neg_g_r_j):
            return np.abs((e_g_y_j - e_g_r_j) - (e_neg_g_y_j - e_neg_g_r_j))

        return FairnessMethods.calculate_average_values(advantaged_group,
                                                        disadvantaged_group,
                                                        number_of_items, predictions,
                                                        ratings, func)

    @staticmethod
    def calculate_abs_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                            advantaged_group: np.array, number_of_items: int):
        def func(e_g_y_j, e_g_r_j, e_neg_g_y_j, e_neg_g_r_j):
            return np.abs(np.abs(e_g_y_j - e_g_r_j) - np.abs(e_neg_g_y_j - e_neg_g_r_j))

        return FairnessMethods.calculate_average_values(advantaged_group,
                                                        disadvantaged_group,
                                                        number_of_items, predictions,
                                                        ratings, func)

    @staticmethod
    def calculate_under_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                              advantaged_group: np.array, number_of_items: int):
        def func(e_g_y_j, e_g_r_j, e_neg_g_y_j, e_neg_g_r_j):
            return np.abs(np.maximum(0, e_g_r_j - e_g_y_j) - np.maximum(0, e_neg_g_r_j - e_neg_g_y_j))

        return FairnessMethods.calculate_average_values(advantaged_group,
                                                        disadvantaged_group,
                                                        number_of_items, predictions,
                                                        ratings, func)

    @staticmethod
    def calculate_over_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                             advantaged_group: np.array, number_of_items: int):
        def func(e_g_y_j, e_g_r_j, e_neg_g_y_j, e_neg_g_r_j):
            return np.abs(np.maximum(0, e_g_y_j - e_g_r_j) - np.maximum(0, e_neg_g_y_j - e_neg_g_r_j))

        return FairnessMethods.calculate_average_values(advantaged_group,
                                                        disadvantaged_group,
                                                        number_of_items, predictions,
                                                        ratings, func)

    @staticmethod
    def calculate_non_parity_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                                   advantaged_group: np.array, number_of_items: int):
        _, e_g_y, _, e_neg_g_y = FairnessMethods.calculate_average_values(advantaged_group,
                                                                          disadvantaged_group,
                                                                          number_of_items, predictions,
                                                                          ratings)
        return np.abs(e_g_y - e_neg_g_y)

    @staticmethod
    def calculate_error_score():
        # TODO:
        return
