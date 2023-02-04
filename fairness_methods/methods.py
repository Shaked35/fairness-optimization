import numpy as np


class FairnessMethods(object):

    @staticmethod
    def calculate_average_values(advantaged_group, disadvantaged_group, number_of_items, predictions, ratings):
        e_g_y = np.zeros(number_of_items)
        e_neg_g_y = np.zeros(number_of_items)
        e_g_r = np.zeros(number_of_items)
        e_neg_g_r = np.zeros(number_of_items)
        for item_pos in range(number_of_items):
            e_g_y[item_pos] = np.mean(predictions[disadvantaged_group][:, item_pos])
            e_neg_g_y[item_pos] = np.mean(predictions[advantaged_group][:, item_pos])
            e_g_r[item_pos] = np.mean(ratings[disadvantaged_group][:, item_pos])
            e_neg_g_r[item_pos] = np.mean(ratings[advantaged_group][:, item_pos])
        return e_g_r, e_g_y, e_neg_g_r, e_neg_g_y

    @staticmethod
    def calculate_val_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                            advantaged_group: np.array, number_of_items: int):
        e_g_r, e_g_y, e_neg_g_r, e_neg_g_y = FairnessMethods.calculate_average_values(advantaged_group,
                                                                                      disadvantaged_group,
                                                                                      number_of_items, predictions,
                                                                                      ratings)

        return np.mean(np.abs((e_g_y - e_g_r) - (e_neg_g_y - e_neg_g_r)))

    @staticmethod
    def calculate_abs_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                            advantaged_group: np.array, number_of_items: int):
        e_g_r, e_g_y, e_neg_g_r, e_neg_g_y = FairnessMethods.calculate_average_values(advantaged_group,
                                                                                      disadvantaged_group,
                                                                                      number_of_items, predictions,
                                                                                      ratings)

        return np.mean(np.abs(np.abs(e_g_y - e_g_r) - np.abs(e_neg_g_y - e_neg_g_r)))

    @staticmethod
    def calculate_under_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                              advantaged_group: np.array, number_of_items: int):
        e_g_r, e_g_y, e_neg_g_r, e_neg_g_y = FairnessMethods.calculate_average_values(advantaged_group,
                                                                                      disadvantaged_group,
                                                                                      number_of_items, predictions,
                                                                                      ratings)

        return np.mean(np.abs(np.maximum(0, e_g_r - e_g_y) - np.maximum(0, e_neg_g_r - e_neg_g_y)))

    @staticmethod
    def calculate_over_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                             advantaged_group: np.array, number_of_items: int):
        e_g_r, e_g_y, e_neg_g_r, e_neg_g_y = FairnessMethods.calculate_average_values(advantaged_group,
                                                                                      disadvantaged_group,
                                                                                      number_of_items, predictions,
                                                                                      ratings)

        return np.mean(np.abs(np.maximum(0, e_g_y - e_g_r) - np.maximum(0, e_neg_g_y - e_neg_g_r)))

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
