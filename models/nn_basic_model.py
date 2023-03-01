import numpy as np
import pandas as pd
import tensorflow as tf

from utils.consts import EPOCH_PRINT


class BasicMatrixFactorizationNN:
    def __init__(self, n_users, n_items, n_factors=20, reg=0.01, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 min_label_value=0, max_label_value=5):
        self.min_label_value = min_label_value
        self.max_label_value = max_label_value
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.reg = reg
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.train_data = None
        # Initialize variables
        self.p = tf.Variable(
            tf.random.normal(shape=(self.n_users, self.n_factors), mean=0.0, stddev=0.01, dtype=tf.float32))
        self.q = tf.Variable(
            tf.random.normal(shape=(self.n_items, self.n_factors), mean=0.0, stddev=0.01, dtype=tf.float32))
        self.u = tf.Variable(tf.zeros(shape=(self.n_users, 1), dtype=tf.float32))
        self.v = tf.Variable(tf.zeros(shape=(self.n_items, 1), dtype=tf.float32))
        print(f"initial model parameters: n_users={n_users}, n_items={n_items}, n_factors={n_factors}, "
              f"reg={reg}, learning_rate={learning_rate}, beta1={beta1}, beta2={beta2}")

    def __call__(self, inputs):
        user_idx, item_idx = inputs[:, 0].astype(np.int64), inputs[:, 1].astype(np.int64)
        user_embed = tf.nn.embedding_lookup(self.p, user_idx)
        item_embed = tf.nn.embedding_lookup(self.q, item_idx)
        user_bias = tf.nn.embedding_lookup(self.u, user_idx)
        item_bias = tf.nn.embedding_lookup(self.v, item_idx)
        dot = tf.reduce_sum(tf.multiply(user_embed, item_embed), axis=1, keepdims=True)
        output = tf.add_n([dot, user_bias, item_bias])
        return output

    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            pred = self(inputs)
            normalized_pred = (pred + 1) / 2
            normalized_pred = normalized_pred * (self.max_label_value - self.min_label_value) + self.min_label_value
            mse_loss = tf.keras.losses.mean_squared_error(labels, normalized_pred)
            reg_loss = tf.nn.l2_loss(self.p) + tf.nn.l2_loss(self.q) + tf.nn.l2_loss(self.u) + tf.nn.l2_loss(self.v)
            total_loss = mse_loss + self.reg * reg_loss
        gradients = tape.gradient(total_loss, [self.p, self.q, self.u, self.v])
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1, beta_2=self.beta2)
        optimizer.apply_gradients(zip(gradients, [self.p, self.q, self.u, self.v]))
        return total_loss.numpy()

    def predict(self, user_id, item_id):
        user_vec = self.p[user_id - 1]
        item_vec = self.q[item_id - 1]
        bias_user = self.u[user_id - 1]
        bias_item = self.v[item_id - 1]
        pred = np.dot(user_vec, item_vec) + bias_user + bias_item
        normalized_pred = (pred + 1) / 2
        return normalized_pred * (self.max_label_value - self.min_label_value) + self.min_label_value

    @staticmethod
    def get_next_batch(train_data, batch_size):
        while True:
            np.random.shuffle(train_data)
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                inputs = batch_data[:, :2]  # extract user and item indices
                labels = batch_data[:, 2:]  # extract ratings
                yield inputs, labels

    def fit(self, train_data, epochs, batch_size, number_of_steps=200):
        loss = []
        pivot_df = pd.DataFrame(train_data)
        self.train_data = pivot_df.melt(ignore_index=False).reset_index()
        self.train_data.columns = ['user_id', 'item_id', 'rating']
        self.train_data = self.train_data.to_numpy()
        for epoch in range(1, epochs + 1):
            for step in range(number_of_steps):
                batch_inputs, batch_labels = self.get_next_batch(self.train_data, batch_size).__next__()
                loss.extend(self.train_step(batch_inputs, batch_labels).tolist())
            if epoch % EPOCH_PRINT == 0:
                print("Epoch: {} Loss: {}".format(epoch, np.mean(loss)))

    def predictions(self):
        inputs = self.train_data[:, :2].copy()
        sorted_indices = np.lexsort((inputs[:, 1], inputs[:, 0]))
        sorted_arr = inputs[sorted_indices]
        predictions = self(sorted_arr)
        normalized_pred = (predictions + 1) / 2
        normalized_pred = normalized_pred * (self.max_label_value - self.min_label_value) + self.min_label_value
        output = pd.DataFrame(inputs, columns=['user_id', 'item_id'])
        output["rating"] = normalized_pred
        return pd.pivot_table(output, values='rating', index='user_id', columns='item_id').to_numpy()
