import numpy as np
import tensorflow as tf
from keras import Model
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
from utils.metrics import RecommendationSystemMetrics
from fairness_methods.methods_tf import FairnessMethods
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout, Multiply, Concatenate

np_config.enable_numpy_behavior()


# Define the value unfairness metric

def get_fairness_metric(metric: str = None, pred=None, ratings=None, dis_group=None, adv_group=None, num_items=None, smooth=True):
    if metric is None:
        fairness = None
    elif metric == 'val_score':
        fairness = FairnessMethods().calculate_val_score(pred, ratings, dis_group, adv_group, num_items, smooth=smooth)
    elif metric == 'abs_score':
        fairness = FairnessMethods().calculate_abs_score(pred, ratings, dis_group, adv_group, num_items, smooth=smooth)
    elif metric == 'over_score':
        fairness = FairnessMethods().calculate_over_score(pred, ratings, dis_group, adv_group, num_items, smooth=smooth)
    elif metric == 'under_score':
        fairness = FairnessMethods().calculate_under_score(pred, ratings, dis_group, adv_group, num_items, smooth=smooth)
    return fairness


# Define the objective function
def objective_function(metric, pred, ratings, dis_group, adv_group, num_items, unfairness_reg=0.5):
    # Compute the value unfairness
    unfairness = get_fairness_metric(metric, pred, ratings, dis_group, adv_group, num_items)
    # Compute the mean squared error loss
    mse_loss = tf.keras.losses.MeanSquaredError()(ratings, pred)
    # Combine the losses
    if unfairness:

        total_loss = mse_loss + unfairness_reg * unfairness
    else:
        total_loss = mse_loss

    return total_loss


# Define the model
def GMF_model_builder(num_users, num_items, embed_dim):
    user_input = layers.Input(shape=(), dtype=tf.float32, name='user_id')
    item_input = layers.Input(shape=(), dtype=tf.float32, name='item_id')
    user_embed = layers.Embedding(num_users, embed_dim)(user_input)
    item_embed = layers.Embedding(num_items, embed_dim)(item_input)
    product = tf.multiply(user_embed, item_embed)
    model = tf.keras.Model(inputs=[user_input, item_input], outputs=product)
    model.summary()
    return model


def NMF_model_builder(num_users, num_items, layers, activation='linear'):
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[-1], name='mf_embedding_user',
                                  input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[-1], name='mf_embedding_item',
                                  input_length=1)

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name="mlp_embedding_user",
                                   input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='mlp_embedding_item',
                                   input_length=1)

    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))

    mf_vector = Multiply()([mf_user_latent, mf_item_latent])
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))

    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
    for index in range(1, len(layers)):
        layer = Dense(layers[index], activation='relu', name=f'layer{index}')
        mlp_vector = layer(mlp_vector)

    predict_vector = Concatenate()([mf_vector, mlp_vector])
    prediction = Dense(1, activation=activation, name="prediction")(predict_vector)
    return Model(inputs=[user_input, item_input], outputs=prediction)


# Train the model using the objective function
def run_nn_fairness_model(model_type, train_set, test_set, num_epochs, num_users, num_items, embed_dim, metric, dis_group, adv_group, lr=0.01,
                          layers=None,
                          early_stop=0.00001, unfairness_reg=0.5, early_stop_tol=1):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_arr = []
    val_score_unfair_arr = []
    val_score_mse_loss_arr = []
    train_score_mse_loss_arr = []
    train_score_unfair_arr = []
    early_stop_counter = 0
    trainig_dict = {}

    if model_type == 'nmf':
        model = NMF_model_builder(num_users, num_items, layers)
    elif model_type == 'gmf':
        model = GMF_model_builder(num_users, num_items, embed_dim)
    else:
        raise NotImplementedError(f'optional models type : nmf/gmf received :{model_type}')

    start_time = datetime.now()

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            # Forward pass
            # Compute the loss
            user_vec = np.vstack([train_set.reset_index()['user_id'].values] * len(train_set.columns)).flatten()
            item_vec = np.vstack([train_set.columns] * len(train_set.index)).flatten()
            pred = model([user_vec, item_vec], training=True)
            pred = pred.reshape(num_users, num_items)
            pred = pred.astype('float32')
            train_set = train_set.astype('float32')
            test_set = test_set.astype('float32')

            loss = objective_function(metric, pred, train_set.values, dis_group, adv_group, num_items, unfairness_reg)
            gradients = tape.gradient(loss, model.trainable_variables)

            # Update the weights
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_arr.append(loss.numpy())
            val_score_mse_loss_arr.append(RecommendationSystemMetrics().RMSE(test_set, pred.numpy()))
            train_score_mse_loss_arr.append(RecommendationSystemMetrics().RMSE(train_set.values, pred.numpy()))
            train_score_unfair = get_fairness_metric(metric, pred, train_set.values, dis_group, adv_group, num_items, False)
            if train_score_unfair is not None:
                train_score_unfair = train_score_unfair.numpy()
            train_score_unfair_arr.append(train_score_unfair)

            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

            if early_stop is not None:
                if len(loss_arr) > 1 and (loss_arr[-2] - loss_arr[-1]) <= early_stop:
                    early_stop_counter += 1
                    if early_stop_counter == early_stop_tol:
                        break

    end_time = datetime.now()
    trainig_dict["model_type"] = model_type
    trainig_dict["fit_time"] = int((end_time - start_time).total_seconds())
    trainig_dict["fit_time"] = int((end_time - start_time).total_seconds())
    trainig_dict["Loss_arr"] = loss_arr
    trainig_dict["metric"] = metric
    trainig_dict["lr"] = lr
    trainig_dict["early_stop"] = early_stop
    trainig_dict["embed_dim"] = embed_dim
    trainig_dict["layers"] = layers
    trainig_dict['model'] = model
    trainig_dict['val_score_unfair_arr'] = val_score_unfair_arr
    trainig_dict['val_score_mse_loss_arr'] = val_score_mse_loss_arr
    trainig_dict['train_score_mse_loss_arr'] = train_score_mse_loss_arr
    trainig_dict['train_score_unfair_arr'] = train_score_unfair_arr

    return trainig_dict
