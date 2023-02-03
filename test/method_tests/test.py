import unittest

import numpy as np

from fairness_methods.methods import FairnessMethods


class MethodTests(unittest.TestCase):
    predictions = np.array([[3, 4, 5, 4.5, 3.3, 2.2], [4, 5, 6.3, 3.5, 3.7, 2.7], [5.3, 6.1, 7.3, 5.5, 4.1, 7.1],
                            [2.3, 2.2, 2.7, 5.2, 3, 4], [3.3, 2.1, 1.4, 5.9, 3, 4.8], [4.6, 2.9, 2.3, 5.4, 3, 4.3]])
    ratings = np.array([[1, 2, 3.4, 4.4, 3.1, 1.5], [2.1, 3, 4.3, 7.3, 6.2, 3.3], [3.4, 4.4, 5.8, 6.6, 7.7, 2.4],
                        [3.5, 4.4, 6.3, 1.3, 2.9, 3.2], [1.5, 5.9, 6.1, 5.6, 3.3, 4.0], [3.6, 4.7, 1.9, 4.6, 3.9, 7.8]])
    disadvantaged_group = [0, 1, 4]
    advantaged_group = [2, 3, 5]
    n = 6

    def test_over(self):
        result = FairnessMethods.calculate_over_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                      self.advantaged_group, self.n)
        print(result)

    def test_under(self):
        result = FairnessMethods.calculate_under_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                       self.advantaged_group, self.n)
        print(result)

    def test_abs(self):
        result = FairnessMethods.calculate_abs_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                     self.advantaged_group, self.n)
        print(result)

    def test_val(self):
        result = FairnessMethods.calculate_val_score(self.predictions, self.ratings, self.disadvantaged_group,
                                                     self.advantaged_group, self.n)
        print(result)
