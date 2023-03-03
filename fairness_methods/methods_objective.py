import numpy as np
from keras.losses import huber_loss
import tensorflow as tf


class FairnessMethods(object):

    @staticmethod
    def calculate_average_values(advantaged_group, disadvantaged_group, number_of_items, predictions, ratings, func):
        values = []
        for item_pos in range(number_of_items):
            e_g_y_j = tf.reduce_mean(predictions[disadvantaged_group][:, item_pos])
            e_neg_g_y_j = tf.reduce_mean(predictions[advantaged_group][:, item_pos])
            e_g_r_j = tf.reduce_mean(ratings[disadvantaged_group][:, item_pos])
            e_neg_g_r_j = tf.reduce_mean(ratings[advantaged_group][:, item_pos])
            values.append(tf.stack([e_g_y_j, e_g_r_j, e_neg_g_y_j, e_neg_g_r_j], axis=0))
        values = tf.stack(values)
        func(tf.gather(values, axis=1, indices=0),
             tf.gather(values, axis=1, indices=1),
             tf.gather(values, axis=1, indices=2),
             tf.gather(values, axis=1, indices=3))
        return tf.reduce_mean(values)

    @staticmethod
    def calculate_val_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                            advantaged_group: np.array, number_of_items: int, smooth=False):
        def func(e_g_y_j, e_g_r_j, e_neg_g_y_j, e_neg_g_r_j):
            if smooth is False:
                return tf.abs(tf.subtract(e_g_y_j, e_g_r_j) - tf.subtract(e_neg_g_y_j, e_neg_g_r_j))
            else:
                return np.abs(huber_loss(e_g_y_j, e_g_r_j) - huber_loss(e_neg_g_y_j, e_neg_g_r_j))

        return FairnessMethods.calculate_average_values(advantaged_group,
                                                        disadvantaged_group,
                                                        number_of_items, predictions,
                                                        ratings, func)

    @staticmethod
    def calculate_abs_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                            advantaged_group: np.array, number_of_items: int, smooth=False):

        def func(e_g_y_j, e_g_r_j, e_neg_g_y_j, e_neg_g_r_j):
            if smooth is False:
                return tf.abs(tf.abs(tf.subtract(e_g_y_j, e_g_r_j)) - tf.abs(tf.subtract(e_neg_g_y_j, e_neg_g_r_j)))
            else:
                return np.abs(tf.abs(huber_loss(e_g_y_j, e_g_r_j)) - tf.abs(huber_loss(e_neg_g_y_j, e_neg_g_r_j)))

        return FairnessMethods.calculate_average_values(advantaged_group,
                                                        disadvantaged_group,
                                                        number_of_items, predictions,
                                                        ratings, func)

    @staticmethod
    def calculate_under_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                              advantaged_group: np.array, number_of_items: int, smooth=False):

        def func(e_g_y_j, e_g_r_j, e_neg_g_y_j, e_neg_g_r_j):
            if smooth is False:
                return tf.abs(tf.maximum(0, tf.subtract(e_g_r_j, e_g_y_j)) - tf.maximum(0, tf.subtract(e_neg_g_r_j, e_neg_g_y_j)))
            else:
                return tf.abs(tf.maximum(0, huber_loss(e_g_r_j, e_g_y_j)) - tf.maximum(0, huber_loss(e_neg_g_r_j, e_neg_g_y_j)))

        return FairnessMethods.calculate_average_values(advantaged_group,
                                                        disadvantaged_group,
                                                        number_of_items, predictions,
                                                        ratings, func)

    @staticmethod
    def calculate_over_score(predictions: np.array, ratings: np.array, disadvantaged_group: np.array,
                             advantaged_group: np.array, number_of_items: int, smooth=False):

        def func(e_g_y_j, e_g_r_j, e_neg_g_y_j, e_neg_g_r_j):
            if smooth is False:
                return tf.abs(tf.maximum(0, tf.subtract(e_g_r_j, e_g_y_j)) - tf.maximum(0, tf.subtract(e_neg_g_y_j, e_neg_g_r_j)))
            else:
                return tf.abs(tf.maximum(0, huber_loss(e_g_r_j, e_g_y_j)) - tf.maximum(0, huber_loss(e_neg_g_y_j, e_neg_g_r_j)))

        return FairnessMethods.calculate_average_values(advantaged_group,
                                                        disadvantaged_group,
                                                        number_of_items, predictions,
                                                        ratings, func)

    @staticmethod
    def calculate_non_parity_score(predictions: np.array, disadvantaged_group: np.array,
                                   advantaged_group: np.array, smooth=False):
        e_g_y = tf.reduce_mean(predictions[disadvantaged_group])
        e_neg_g_y = tf.reduce_mean(predictions[advantaged_group])

        if smooth is False:
            return tf.abs(e_g_y - e_neg_g_y)
        else:
            return tf.abs(huber_loss(e_g_y, e_neg_g_y))

    @staticmethod
    def calculate_error_score(predictions: np.array, ratings: np.array):

        inx = ratings > 0
        mse = np.sqrt(np.mean((ratings[inx] - predictions[inx]) ** 2))
        return mse
