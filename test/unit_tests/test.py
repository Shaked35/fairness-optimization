import unittest

from tabulate import tabulate

from fairness_methods.methods import FairnessMethods
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

    headers = ["item 1", "item 2", "item 3", "item 4", "item 5", "item 6"]

    # Generate the table in fancy format.
    table_predictions = tabulate(predictions, headers, tablefmt="fancy_grid")
    table_ratings = tabulate(ratings, headers, tablefmt="fancy_grid")

    print("****************************predictions****************************\n")
    print(table_predictions)

    print("\n****************************ratings****************************\n")
    print(table_ratings)

    def test_over(self):
        expected = 0.45
        result = FairnessMethods.calculate_over_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                      self.advantaged_group, self.n)
        assert round(result, 2) == expected, f"Expected {expected} but got {result}"

    def test_under(self):
        expected = 0.43
        result = FairnessMethods.calculate_under_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                       self.advantaged_group, self.n)
        assert round(result, 2) == expected, f"Expected {expected} but got {result}"

    def test_abs(self):
        expected = 0.55
        result = FairnessMethods.calculate_abs_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                     self.advantaged_group, self.n)
        assert round(result, 2) == expected, f"Expected {expected} but got {result}"

    def test_val(self):
        expected = 0.88
        result = FairnessMethods.calculate_val_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                     self.advantaged_group, self.n)
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
        assert r.shape == (n_users, n_items), f"Expected shape {(n_users, n_items)}, but got {r.shape}"
        assert np.abs(np.sum(r == 1) / np.size(r) - np.mean(
            L * O)) < 0.1, "Expected approximately equal distribution of +1 and -1 ratings."

    def test_real_data(self):
        r = generate_real_data()
        print(r.head())
