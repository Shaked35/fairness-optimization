import numpy as np
import pandas as pd


class RecommendationSystemMetrics(object):

    def check_numpy_arr(self, true_array, pred_array):

        if not isinstance(true_array, np.ndarray):
            true_array = true_array.to_numpy()

        if not isinstance(pred_array, np.ndarray):
            pred_array = pred_array.to_numpy()

        return true_array, pred_array

    def _hendle_edge_cases(self, true_array: np.ndarray, pred_array: np.ndarray, lower_bound: int = 1,
                           upper_bound: int = 5):
        true_array, pred_array = self.check_numpy_arr(true_array, pred_array)
        mask = ~(true_array == 0)
        true_array, pred_array = true_array[mask], pred_array[mask]
        pred_array = np.clip(pred_array, lower_bound, upper_bound)

        return true_array, pred_array

    def RMSE(self, true_array: np.ndarray, pred_array: np.ndarray, lower_bound: int = 1, upper_bound: int = 5):

        return np.sqrt(np.nanmean((true_array.flatten() - pred_array.flatten()) ** 2))

    def MRR(self, df_true, df_pred, lower_bound=1, upper_bound=5, th: int = 3, top_n=5):

        df_true, df_pred = self.check_numpy_arr(df_true, df_pred)
        sum_, n = 0, df_true.shape[0]
        counter = 0

        for user_row in range(0, n):  # loop over all users
            user_true, user_pred = df_true[user_row, :], df_pred[user_row, :]
            best_idx = np.argsort(user_pred)[::-1][:top_n]
            user_true, user_pred = user_true[best_idx], user_pred[best_idx]
            user_true, user_pred = self._hendle_edge_cases(user_true, user_pred, lower_bound, upper_bound)
            if sum(user_true) > 0:
                sum_ += self.MRR_for_user(user_true, user_pred, th=th)
            else:
                counter += 1
        if n - counter != 0:
            return sum_ / (n - counter)
        else:
            return 0

    def MRR_for_user(self, user_true, user_pred, th=3):
        # RR(u)= sum (relvance/rank) for i> th)
        df = pd.DataFrame({"true": user_true, "pred": user_pred}) >= th  # bool df with true and pred >= th
        df['true_pred_sum'] = df.sum(axis=1)  # add sum column
        df = df.sort_values(by='true_pred_sum', ascending=False)  # sort values by true_pred_sum
        if df.true_pred_sum.reset_index(drop=True)[0] == 2:  # if first sum ==2 return its inx +1
            return 1 / (df.index[0] + 1)
        return 0  # else return 0

    def NDCG(self, df_true, df_pred, lower_bound=1, upper_bound=5, top_n=5):
        df_true, df_pred = self.check_numpy_arr(df_true, df_pred)
        sum_, n, counter = 0, df_true.shape[0], 0
        for user_row in range(0, n):  # loop over all users
            user_true, user_pred = df_true[user_row, :], df_pred[user_row, :]
            best_idx = np.argsort(user_pred)[::-1][:top_n]
            user_pred = user_true[best_idx]
            user_true = user_true[best_idx]
            user_true = np.sort(user_true)[::-1]
            if sum(user_true) > 0:
                user_true, user_pred = self._hendle_edge_cases(user_true, user_pred, lower_bound, upper_bound)
                sum_ += self.NDCG_for_user(user_true, user_pred)
            else:
                counter += 1
        if n - counter != 0:
            return sum_ / (n - counter)
        else:
            return 0

    def NDCG_for_user(self, user_true, user_pred):
        idcg = self.DCG(user_true)
        dcg = self.DCG(user_pred)
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0
        return ndcg

    def DCG(self, rel):
        return sum(rel / np.log2(np.arange(len(rel)) + 2))

    def get_error(self, df_true, df_pred):
        return {'RMSE': self.RMSE(df_true, df_pred), 'MRR_5': self.MRR(df_true, df_pred, top_n=5),
                'MRR_10': self.MRR(df_true, df_pred, top_n=10), 'NDCG_5': self.NDCG(df_true, df_pred, top_n=5),
                'NDCG_10': self.NDCG(df_true, df_pred, top_n=10)}
