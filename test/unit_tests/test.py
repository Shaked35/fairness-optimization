import random
import unittest

import tensorflow

from fairness_methods.methods import FairnessMethods
from models.basic_mf_model import BasicMatrixFactorization
from utils.data_generator import *


class MethodTests(unittest.TestCase):
    predictions = np.array([[3, 4, 5, 4.5, 3.3, 2.2], [4, 5, 6.3, 3.5, 3.7, 2.7], [5.3, 6.1, 7.3, 5.5, 4.1, 7.1],
                            [2.3, 2.2, 2.7, 5.2, 3, 4], [3.3, 2.1, 1.4, 5.9, 3, 4.8], [4.6, 2.9, 2.3, 5.4, 3, 4.3],
                            [3.6, 4.2, 1.4, 4.1, 3.5, 6.8]])
    ratings = np.array([[1, 2, 3.4, 4.4, 3.1, 1.5], [2.1, 3, 4.3, 7.3, 6.2, 3.3], [3.4, 4.4, 5.8, 6.6, 7.7, 2.4],
                        [3.5, 4.4, 6.3, 1.3, 2.9, 3.2], [1.5, 5.9, 6.1, 5.6, 3.3, 4.0], [3.6, 4.7, 1.9, 4.6, 3.9, 7.8],
                        [3.6, 4.7, 1.9, 4.6, 3.9, 7.8]])
    disadvantaged_group = [0, 1, 4, 6]
    advantaged_group = [2, 3, 5]
    n = 6

    def print_table(self):
        headers = ["item 1", "item 2", "item 3", "item 4", "item 5", "item 6"]
        users = ["user 1", "user 2", "user 3", "user 4", "user 5", "user 6", "user 7"]
        # Generate the table in fancy format.
        table_predictions = tabulate(pd.DataFrame(self.predictions, columns=headers, index=users),
                                     headers=headers, showindex=True, tablefmt="fancy_grid")
        table_ratings = tabulate(pd.DataFrame(self.ratings, columns=headers, index=users),
                                 headers=headers, showindex=True, tablefmt="fancy_grid")

        print("****************************predictions****************************\n")
        print(table_predictions)

        print("\n****************************ratings****************************\n")
        print(table_ratings)

    def test_over(self):
        self.print_table()
        expected = 0.45
        result = FairnessMethods.calculate_over_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                      self.advantaged_group, self.n)
        assert round(result, 2) == expected, f"Expected {expected} but got {result}"

    def test_under(self):
        self.print_table()
        expected = 0.43
        result = FairnessMethods.calculate_under_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                       self.advantaged_group, self.n)
        assert round(result, 2) == expected, f"Expected {expected} but got {result}"

    def test_abs(self):
        self.print_table()
        expected = 0.55
        result = FairnessMethods.calculate_abs_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                     self.advantaged_group, self.n)
        assert round(result, 2) == expected, f"Expected {expected} but got {result}"

    def test_val(self):
        self.print_table()
        expected = 0.88
        result = FairnessMethods.calculate_val_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                     self.advantaged_group, self.n)
        assert round(result, 2) == expected, f"Expected {expected} but got {result}"

    def test_par(self):
        self.print_table()
        expected = 0.49
        result = FairnessMethods.calculate_non_parity_score(self.predictions, self.disadvantaged_group,
                                                            self.advantaged_group)
        assert round(result, 2) == expected, f"Expected {expected} but got {result}"

    def test_synthetic_data(self):
        L = np.array([[0.8, 0.2, 0.2], [0.8, 0.8, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8]])
        O = np.array([[0.6, 0.2, 0.1], [0.3, 0.4, 0.2], [0.1, 0.3, 0.5], [0.05, 0.5, 0.35]])
        n_users = 400
        n_items = 300
        headers = ["Fem", "STEM", "Masc"]
        table_L = tabulate(L, headers, tablefmt="fancy_grid")
        print(table_L)
        r = generate_synthetic_data(L, O, num_users=n_users, num_items=n_items)
        assert len(r[0]) == n_users * n_items, f"Expected shape {(n_users, n_items)}, but got {r.shape}"
        assert np.abs(np.sum(r[0]["rating"] == 1) / len(r[0]) - np.mean(
            L * O)) < 0.1, "Expected approximately equal distribution of +1 and -3 ratings."

    def test_generate_synthetic_data(self):
        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate synthetic data with uniform observation probability and balanced user group distribution
        rating_df, items = generate_synthetic_data(observation_model='unbalanced', user_distribution='balanced')

        # Check that the shape of the data is as expected
        ratings, observations = data[0], data[1]

        assert ratings.shape == (400, 300)
        assert observations.shape == (400, 300)

        # Check that the rating values are either +1 or -1
        assert np.all(np.isin(ratings, [-1, 1]))

        # Check that the observation values are either 0 or 1
        assert np.all(np.isin(observations, [0, 1]))

        # Generate synthetic data with unbalanced observation probability and imbalanced user group distribution
        data = generate_synthetic_data(observation_model='uniform', user_distribution='imbalanced')
        ratings, observations = data[0], data[1]
        # Check that the shape of the data is as expected
        assert ratings.shape == (400, 300)
        assert observations.shape == (400, 300)

        # Check that the rating values are either +1 or -1
        assert np.all(np.isin(ratings, [-1, 1]))

        # Check that the observation values are either 0 or 1
        assert np.all(np.isin(observations, [0, 1]))

    @staticmethod
    def get_next_batch(train_data, batch_size):
        np.random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            inputs = batch_data[:, :2]  # extract user and item indices
            labels = batch_data[:, 2:]  # extract ratings
            yield inputs, labels

    def test_model(self):
        # Generate sample data
        n_users = 10
        n_items = 6
        n_groups = 2
        X = np.random.randint(1, 5, size=(n_users, n_items))
        group_user = {id: random.randint(0, n_groups - 1) for id in range(n_users)}
        group_item = {id: random.randint(0, n_groups - 1) for id in range(n_items)}
        model = BasicMatrixFactorization(n_users, n_items, n_factors=2)
        x1 = np.repeat(np.arange(X.shape[0]), len(X.flatten()) / len(np.arange(X.shape[0])))
        x2 = np.tile(np.arange(X.shape[1]), int(len(X.flatten()) / len(np.arange(X.shape[1]))))
        x3 = X.flatten()

        train_data = pd.DataFrame(np.array([x1, x2, x3]).T, columns=['X', 'Y', 'Z'])
        model.fit(X, group_user, group_item)
        pred = model.predictions()
        assert pred.shape == (n_users, n_items)
