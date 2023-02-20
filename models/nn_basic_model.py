import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot
from tensorflow.keras.optimizers import Adam

from utils.consts import DEFAULT_LAMBDA


class NNBasicMF(object):
    def __init__(self, input_pivot_table, embedding_size=10, learning_rate=DEFAULT_LAMBDA):
        self.ratings_df = input_pivot_table.stack().reset_index()
        self.ratings_df.columns = ['user', 'item', 'rating']

        # Define the number of users, items, and embedding size
        num_users = input_pivot_table.shape[0]
        num_items = input_pivot_table.shape[1]

        # Define the inputs and embeddings for users and items
        user_input = Input(shape=(1,))
        user_embedding = Embedding(num_users, embedding_size)(user_input)
        user_flattened = Flatten()(user_embedding)

        item_input = Input(shape=(1,))
        item_embedding = Embedding(num_items, embedding_size)(item_input)
        item_flattened = Flatten()(item_embedding)

        # Define the dot product and output layer
        dot_product = Dot(axes=1)([user_flattened, item_flattened])
        output_layer = dot_product

        # Define the model and compile it with the Adam optimizer and MSE loss
        self.model = Model(inputs=[user_input, item_input], outputs=output_layer)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        # Convert the ratings DataFrame to arrays for training
        self.user_indices = self.ratings_df['user'].values
        self.item_indices = self.ratings_df['item'].values
        self.ratings = self.ratings_df['rating'].values
        # Normalize the item indices so that they start from 0
        self.item_indices -= self.item_indices.min()
        self.user_indices -= self.user_indices.min()

    def fit(self, batch_size=2, epochs=10):
        self.model.fit([self.user_indices, self.item_indices], self.ratings, batch_size=batch_size, epochs=epochs)

    def predict(self, minimum_output_value=0, maximum_output_value=5, normalize=False):
        predictions = pd.DataFrame(self.model.predict([self.user_indices, self.item_indices]), columns=["rating"])
        if normalize:
            predictions["rating"] = NNBasicMF.normalized_values(list(predictions["rating"]), minimum_output_value,
                                                              maximum_output_value)
        predictions["user"] = self.user_indices
        predictions["item"] = self.item_indices
        results = pd.pivot_table(predictions, values='rating', index='user', columns="item")
        results.fillna(0, inplace=True)
        return results.to_numpy()

    def get_rmse(self):
        predictions = self.model.predict([self.user_indices, self.item_indices])
        return np.sqrt(((predictions - self.ratings) ** 2).mean())

    @staticmethod
    def normalized_values(input_list, minimum_output_value, maximum_output_value):
        # find the minimum and maximum values in the input list
        input_min = min(input_list)
        input_max = max(input_list)

        # create an empty list to hold the normalized values
        normalized_list = []

        # loop over each value in the input list
        for value in input_list:
            # normalize the value and add it to the normalized list
            normalized_value = ((value - input_min) / (input_max - input_min)) * (
                    maximum_output_value - minimum_output_value) + minimum_output_value
            normalized_list.append(normalized_value)

        return normalized_list
