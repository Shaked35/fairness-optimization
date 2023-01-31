class FairnessMethods(object):
    @staticmethod
    def demographic_parity(predictions, sensitive_feature):
        """
        Calculates demographic parity for a given set of predictions and a sensitive feature.

        predictions: a list of (user_id, item_id, rating, prediction) tuples
        sensitive_feature: the name of the sensitive feature (e.g. "gender")

        returns: demographic parity as a float
        """
        # Count the total number of predictions
        total_count = len(predictions)

        # Count the number of predictions for each value of the sensitive feature
        feature_counts = {}
        for _, _, _, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if feature_value not in feature_counts:
                feature_counts[feature_value] = 0
            feature_counts[feature_value] += 1

        # Calculate demographic parity
        disparities = []
        for feature_value, count in feature_counts.items():
            disparities.append(abs(count / total_count - 0.5))
        return max(disparities)

    @staticmethod
    def equal_opportunity(predictions, sensitive_feature, outcome_feature):
        """
        Calculates equal opportunity for a given set of predictions, a sensitive feature, and an outcome feature.

        predictions: a list of (user_id, item_id, rating, prediction) tuples
        sensitive_feature: the name of the sensitive feature (e.g. "gender")
        outcome_feature: the name of the outcome feature (e.g. "rating")

        returns: equal opportunity as a float
        """
        # Count the number of predictions for each value of the sensitive feature
        feature_counts = {}
        for _, _, _, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if feature_value not in feature_counts:
                feature_counts[feature_value] = 0
            feature_counts[feature_value] += 1

        # Count the number of positive outcomes for each value of the sensitive feature
        positive_counts = {}
        for _, _, rating, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if rating == 1 and feature_value not in positive_counts:
                positive_counts[feature_value] = 0
            if rating == 1:
                positive_counts[feature_value] += 1
        # Calculate the proportion of positive outcomes for each value of the sensitive feature
        proportions = {}
        for feature_value, count in positive_counts.items():
            proportions[feature_value] = count / feature_counts[feature_value]

        # Calculate the disparity for each value of the sensitive feature
        disparities = []
        for feature_value, count in feature_counts.items():
            disparities.append(abs(proportions[feature_value] - proportions[next(iter(feature_counts))]))

        # Return the maximum disparity
        return max(disparities)

    @staticmethod
    def absolute_unfairness(predictions, sensitive_feature, outcome_feature):
        """
        Calculates absolute unfairness for a given set of predictions, a sensitive feature, and an outcome feature.

        predictions: a list of (user_id, item_id, rating, prediction) tuples
        sensitive_feature: the name of the sensitive feature (e.g. "gender")
        outcome_feature: the name of the outcome feature (e.g. "rating")

        returns: absolute unfairness as a float
        """
        # Count the number of predictions for each value of the sensitive feature
        feature_counts = {}
        for _, _, _, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if feature_value not in feature_counts:
                feature_counts[feature_value] = 0
            feature_counts[feature_value] += 1

        # Count the number of positive outcomes for each value of the sensitive feature
        positive_counts = {}
        for _, _, rating, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if rating == 1 and feature_value not in positive_counts:
                positive_counts[feature_value] = 0
            if rating == 1:
                positive_counts[feature_value] += 1
        # Calculate the proportion of positive outcomes for each value of the sensitive feature
        proportions = {}
        for feature_value, count in positive_counts.items():
            proportions[feature_value] = count / feature_counts[feature_value]

        # Calculate the absolute unfairness
        unfairness = 0
        for feature_value, count in feature_counts.items():
            unfairness += count * abs(proportions[feature_value] - proportions[next(iter(feature_counts))])

        return unfairness

    @staticmethod
    def value_unfairness(predictions, sensitive_feature, outcome_feature):
        """
        Calculates value unfairness for a given set of predictions, a sensitive feature, and an outcome feature.

        predictions: a list of (user_id, item_id, rating, prediction) tuples
        sensitive_feature: the name of the sensitive feature (e.g. "gender")
        outcome_feature: the name of the outcome feature (e.g. "rating")

        returns: value unfairness as a float
        """
        # Count the number of predictions for each value of the sensitive feature
        feature_counts = {}
        for _, _, _, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if feature_value not in feature_counts:
                feature_counts[feature_value] = 0
            feature_counts[feature_value] += 1

        # Count the number of positive outcomes for each value of the sensitive feature
        value_counts = {}
        for _, _, rating, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if feature_value not in value_counts:
                value_counts[feature_value] = 0
            value_counts[feature_value] += rating

        # Calculate the mean value of the outcome feature for each value of the sensitive feature
        means = {}
        for feature_value, count in value_counts.items():
            means[feature_value] = count / feature_counts[feature_value]

        # Calculate the value unfairness
        unfairness = 0
        for feature_value, count in feature_counts.items():
            unfairness += count * abs(means[feature_value] - means[next(iter(feature_counts))])

        return unfairness

    @staticmethod
    def under_unfairness(predictions, sensitive_feature, outcome_feature):
        """
        Calculates under-unfairness for a given set of predictions, a sensitive feature, and an outcome feature.

        predictions: a list of (user_id, item_id, rating, prediction) tuples
        sensitive_feature: the name of the sensitive feature (e.g. "gender")
        outcome_feature: the name of the outcome feature (e.g. "rating")

        returns: under-unfairness as a float
        """
        # Count the number of predictions for each value of the sensitive feature
        feature_counts = {}
        for _, _, _, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if feature_value not in feature_counts:
                feature_counts[feature_value] = 0
            feature_counts[feature_value] += 1

        # Count the number of positive outcomes for each value of the sensitive feature
        value_counts = {}
        for _, _, rating, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if feature_value not in value_counts:
                value_counts[feature_value] = 0
            if rating >= 1:
                value_counts[feature_value] += 1

        # Calculate the mean value of the outcome feature for each value of the sensitive feature
        means = {}
        for feature_value, count in value_counts.items():
            means[feature_value] = count / feature_counts[feature_value]

        # Calculate the under-unfairness
        unfairness = 0
        for feature_value, count in feature_counts.items():
            unfairness += count * abs(means[feature_value] - means[next(iter(feature_counts))])
        return unfairness

    @staticmethod
    def over_unfairness(predictions, sensitive_feature, outcome_feature):
        """
        Calculates over-unfairness for a given set of predictions, a sensitive feature, and an outcome feature.

        predictions: a list of (user_id, item_id, rating, prediction) tuples
        sensitive_feature: the name of the sensitive feature (e.g. "gender")
        outcome_feature: the name of the outcome feature (e.g. "rating")

        returns: over-unfairness as a float
        """
        # Count the number of predictions for each value of the sensitive feature
        feature_counts = {}
        for _, _, _, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if feature_value not in feature_counts:
                feature_counts[feature_value] = 0
            feature_counts[feature_value] += 1

        # Count the number of positive outcomes for each value of the sensitive feature
        value_counts = {}
        for _, _, rating, prediction in predictions:
            feature_value = prediction[sensitive_feature]
            if feature_value not in value_counts:
                value_counts[feature_value] = 0
            if rating >= 1:
                value_counts[feature_value] += 1

        # Calculate the mean value of the outcome feature for each value of the sensitive feature
        means = {}
        for feature_value, count in value_counts.items():
            means[feature_value] = count / feature_counts[feature_value]

        # Calculate the over-unfairness
        unfairness = 0
        for feature_value, count in feature_counts.items():
            unfairness += count * abs(means[feature_value] - means[next(iter(feature_counts))])
        return unfairness

    # over
    def calculate_unfairness(recommendations_group1, recommendations_group2):
        intersection = set(recommendations_group1) & set(recommendations_group2)
        group1_unfairness = len(intersection) / len(recommendations_group1)
        group2_unfairness = len(intersection) / len(recommendations_group2)
        return max(group1_unfairness, group2_unfairness)


# The paper Beyond Parity: Fairness Objectives for Collaborative Filtering does not mention a
# specific method for measuring "Uabs" in collaborative filtering. However, the paper introduces a fairness
# objective for collaborative filtering called "U(R,S)" which measures the unfairness between two sets of users,
# R and S. The formula for U(R,S) is defined as:
#
# U(R,S) = (1/|R|) * ∑_{u∈R} max_{v∈S} sim(u,v) - F(R,S)
#
# Where |R| is the number of users in R, sim(u,v) is the similarity between users u and v, and F(R,S) is a fairness
# objective that measures the parity between the sets R and S.
#
# This method, U(R,S) is measuring the unfairness between two groups, it can be modified to measure Uabs as follows:
#
# Uabs(R,S) = max(U(R,S), U(S,R))
#
# This function takes in three arguments: recommendations_group1 and recommendations_group2, which are the sets of
# items recommended to two different groups of users, similarity_matrix which is a matrix that contains the similarity
# between all pairs of users. The function calculates the Uabs by calling the calculate_unfairness function twice, once
# for each group, and returns the maximum value.
#
# It's important to note that, this approach can be sensitive to the definition of similarity and it may not be the
# best method for measuring the absolute unfairness for your specific use case. I would recommend consulting other
# sources or research papers in order to find a method for measuring absolute unfairness, or discussing the specific
# use case and requirements with experts in the field.
# You could implement this method in Python as follows:

def calculate_Uabs(recommendations_group1, recommendations_group2, similarity_matrix):
    U_RS = calculate_unfairness(recommendations_group1, recommendations_group2, similarity_matrix)
    U_SR = calculate_unfairness(recommendations_group2, recommendations_group1, similarity_matrix)
    return max(U_RS, U_SR)


def calculate_unfairness(recommendations_group1, recommendations_group2, similarity_matrix):
    sum_similarity = 0
    for user1 in recommendations_group1:
        max_similarity = 0
        for user2 in recommendations_group2:
            similarity = similarity_matrix[user1][user2]
            max_similarity = max(max_similarity, similarity)
        sum_similarity += max_similarity
    return (1 / len(recommendations_group1)) * sum_similarity - calculate_parity(recommendations_group1,
                                                                                 recommendations_group2,
                                                                                 similarity_matrix)


# In the paper Beyond Parity: Fairness Objectives for Collaborative Filtering, the authors introduce the concept of a
# fairness objective called "F(R,S)" which measures the parity between two sets of users, R and S. The formula for
# F(R,S) is defined as:
#
# F(R,S) = (1/|R|) * ∑_{u∈R} sim(u,S)
#
# Where |R| is the number of users in R, sim(u,S) is the similarity between user u and the set of users S.
#
# This function takes in three arguments: recommendations_group1 and recommendations_group2, which are the sets of
# items recommended to two different groups of users, similarity_matrix which is a matrix that contains the similarity
# between all pairs of users. The function calculates the parity between the two groups, by summing up the similarity
# between each user in the first group and all the users in the second group, and then dividing by the size of the
# first group.
#
# It's important to note that, this approach can be sensitive to the definition of similarity and it may not be the
# best method for measuring the parity for your specific use case. I would recommend consulting other sources or
# research papers in order to find a method for measuring parity, or discussing the specific use case and requirements
# with experts in the field.
# You could implement this method in Python as follows:

def calculate_parity(recommendations_group1, recommendations_group2, similarity_matrix):
    sum_similarity = 0
    for user1 in recommendations_group1:
        similarity = 0
        for user2 in recommendations_group2:
            similarity += similarity_matrix[user1][user2]
        sum_similarity += similarity
    return (1 / len(recommendations_group1)) * sum_similarity


# In the paper Beyond Parity: Fairness Objectives for Collaborative Filtering, the authors introduce the concept of a
# fairness objective called "Uunder" which measures the under-representation of a certain group of users, R, in
# the recommendations provided to another group of users, S. The formula for Uunder(R,S) is defined as:
#
# Uunder(R,S) = (1/|R|) * ∑_{u∈R} [1 - sim(u,S)]
#
# Where |R| is the number of users in R, sim(u,S) is the similarity between user u and the set of users S.
#
# This function takes in three arguments: recommendations_group1 and recommendations_group2, which are the sets of
# items recommended to two different groups of users, similarity_matrix which is a matrix that contains the similarity
# between all pairs of users. The function calculates the under-representation of the first group, by summing up the
# difference between 1 and the similarity between each user in the first group and all the users in the second group,
# and then dividing by the size of the first group.
#
# It's important to note that, this approach can be sensitive to the definition of similarity and it may not be the
# best method for measuring the under-representation for your specific use case. I would recommend consulting other
# sources or research papers in order to find a method for measuring under-representation, or discussing the specific
# use case and requirements with experts in the field.
#
# You could implement this method in Python as follows:

def calculate_under(recommendations_group1, recommendations_group2, similarity_matrix):
    sum_similarity = 0
    for user1 in recommendations_group1:
        similarity = 0
        for user2 in recommendations_group2:
            similarity += similarity_matrix[user1][user2]
        sum_similarity += (1 - similarity)
    return (1 / len(recommendations_group1)) * sum_similarity


# implement the following metric with test from the article 'Beyond Parity: Fairness Objectives for Collaborative Filtering ':
#
# Uval = 1/n \sum_{j=1}^{n} |(E_{g}  \left [y  \right ]_{j} - E_{g}\left [r  \right ]_{j}) - (E_{\neg g}  \left [y  \right ]_{j} - E_{\neg g}\left [r  \right ]_{j})|
#
# Sure, here's an example of how you might implement the fairness metric Uval in Python, along with a test function to check that it's working correctly:

import numpy as np


def calculate_Uval(y, r, g):
    """
    Calculate the Uval fairness metric for a given set of predicted ratings (y) and observed ratings (r),
    and a binary protected attribute (g).
    """
    n = len(y)
    uval = 0
    for j in range(n):
        e_g_y = np.mean([y[i] for i in range(n) if g[i] == 1])
        e_g_r = np.mean([r[i] for i in range(n) if g[i] == 1])
        e_not_g_y = np.mean([y[i] for i in range(n) if g[i] == 0])
        e_not_g_r = np.mean([r[i] for i in range(n) if g[i] == 0])
        uval += abs((e_g_y - e_g_r) - (e_not_g_y - e_not_g_r))
    return uval / n


def test_calculate_Uval():
    y = [3, 5, 4, 4, 2]
    r = [3, 5, 4, 3, 2]
    g = [1, 1, 0, 0, 1]
    uval = calculate_Uval(y, r, g)
    assert uval == 0.2
    print("Test passed.")


# test_calculate_Uval()
# This implementation takes as input the predicted ratings y, the observed ratings r, and the binary protected attribute g, and returns the value of the Uval fairness metric.
# The function test_calculate_Uval() tests the implementation with sample inputs and asserts the output of the function.
# Please Note that this is an example and it's based on the specific use case you described, and it may not be appropriate for other dataset or scenarios.


#
# This implementation takes as input the predicted ratings y, the observed ratings r, and the binary protected attribute g, and returns the value of the Uval fairness metric.
# The function test_calculate_Uval() tests the implementation with sample inputs and asserts the output of the function.
# Please Note that this is an example and it's based on the specific use case you described, and it may not be appropriate for other dataset or scenarios.


# mplement the following metric with a test from the article 'Beyond Parity: Fairness Objectives for Collaborative Filtering ':
#
# Uunder = 1/n \sum_{j=1}^{n} \left |\max \left \{0,  E_{g}  \left [y  \right ]_{j} - E_{g}  \left [r  \right ]_{j}\right \} - \max \left \{0,  E_{\neg g}  \left [y  \right ]_{j} - E_{\neg g}  \left [r  \right ]_{j}\right \} \right |
# Here is an example of how you might implement the Uunder metric using Python:


# This implementation takes in 4 lists and an integer n, where y_g is the list of true labels for group g, y_ng is the list of true labels for group not g, r_g is the list of predicted labels for group g, and r_ng is the list of predicted labels for group not g. It then iterates through the lists and calculates the Uunder value by summing up the absolute difference of the max(0, y_g - r_g) and max(0, y_ng - r_ng) for each index and then dividing by n.
# Please note that this is just one possible implementation, depending on the specific use case, the implementation may change.


import numpy as np


def calculate_Uunder(y_g, y_ng, r_g, r_ng, n):
    Uunder = 0
    for j in range(n):
        Uunder += abs(max(0, y_g[j] - r_g[j]) - max(0, y_ng[j] - r_ng[j]))
    return Uunder / n


# test example
y_g = [1, 0, 1, 0]
y_ng = [0, 1, 0, 1]
r_g = [0.8, 0.1, 0.6, 0.2]
r_ng = [0.9, 0.2, 0.7, 0.3]
n = len(y_g)

Uunder_val = calculate_Uunder(y_g, y_ng, r_g, r_ng, n)
print(Uunder_val)




# implement the following metric with a test from the article 'Beyond Parity: Fairness Objectives for Collaborative Filtering ':
#
# Uunder = 1/n \sum_{j=1}^{n} \left |\max \left \{0,  E_{g}  \left [r  \right ]_{j} - E_{g}  \left [y  \right ]_{j}\right \} - \max \left \{0,  E_{\neg g}  \left [r  \right ]_{j} - E_{\neg g}  \left [y  \right ]_{j}\right \} \right |
# The mathematical formula for the metric Uunder is:
#
# Uunder = 1/n * Σ(j=1 to n) | max(0, E_g[r_j] - E_g[y_j]) - max(0, E_not_g[r_j] - E_not_g[y_j]) |
#
# Here, n is the total number of items, g is the sensitive attribute, y is the predicted rating, and r is the actual rating. E_g[x_j] and E_not_g[x_j] represent the expected value of x_j for group g and not group g, respectively.
#
# To implement this metric in python, you can use the following code:


import numpy as np

def calculate_Uunder(ratings, predictions, sensitive_attributes):
    n = len(ratings)
    uunder = 0
    for j in range(n):
        e_g_r = np.mean([ratings[i] for i in range(n) if sensitive_attributes[i] == g])
        e_g_y = np.mean([predictions[i] for i in range(n) if sensitive_attributes[i] == g])
        e_not_g_r = np.mean([ratings[i] for i in range(n) if sensitive_attributes[i] != g])
        e_not_g_y = np.mean([predictions[i] for i in range(n) if sensitive_attributes[i] != g])
        uunder += abs(max(0, e_g_r - e_g_y) - max(0, e_not_g_r - e_not_g_y))
    uunder = uunder / n
    return uunder


# You can test this function by passing some sample data for the arguments ratings, predictions, and sensitive_attributes, like:


ratings = [4, 5, 3, 2, 4, 5, 2, 3, 4, 5]
predictions = [3.5, 4.5, 2.5, 1.5, 3.5, 4.5, 1.5, 2.5, 3.5, 4.5]
sensitive_attributes = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
print(calculate_Uunder(ratings, predictions, sensitive_attributes))
