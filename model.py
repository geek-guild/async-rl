import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, BatchNormalization, Flatten, Dense, Input
from keras.models import Model

def build_network(num_actions, agent_history_length, model_type='dnn', resized_width=None, resized_height=None, input_size=None, name_scope=None, layers=26, hidden_width=256):
    model_type = model_type or 'dnn'
    if model_type == 'dnn':
        return build_dnn_network(num_actions, agent_history_length, input_size, name_scope, layers, hidden_width)
    elif model_type == '2d_cnn':
        return build_2d_cnn_network(num_actions, agent_history_length, resized_width, resized_height, name_scope)


def build_dnn_network(num_actions, agent_history_length, input_size, name_scope, layers, hidden_width):
    with tf.device("/cpu:0"):
        with tf.name_scope(name_scope):
            state = tf.placeholder(tf.float32, [None, agent_history_length, input_size],
                                   name="state")
            inputs = Input(shape=(agent_history_length, input_size,))
            # model = Flatten()(inputs)
            model = inputs
            for l in range(layers):
                name = "h{}".format(l)
                model = Dense(name=name, output_dim=hidden_width, activation='relu')(model)
                model = BatchNormalization()(model)

            model = Flatten()(model)
            model = Dense(256, activation='relu')(model)
            print(model)
            q_values = Dense(num_actions)(model)

            # UserWarning: Update your `Model` call to the Keras 2 API:
            # `Model(outputs=Tensor("de..., inputs=Tensor("in..
            m = Model(inputs=inputs, outputs=q_values)

    return state, m

def build_2d_cnn_network(num_actions, agent_history_length, resized_width, resized_height, name_scope):
  with tf.device("/cpu:0"):
    with tf.name_scope(name_scope):
        state = tf.placeholder(tf.float32, [None, agent_history_length, resized_width, resized_height], name="state")
        inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
        model = Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), activation='relu', padding='same', data_format='channels_first')(inputs)
        model = Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same', data_format='channels_first')(model)
        #model = Conv2D(filter=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(model)
        model = Flatten()(model)
        model = Dense(256, activation='relu')(model)
        print(model)
        q_values = Dense(num_actions)(model)
        
        #UserWarning: Update your `Model` call to the Keras 2 API: 
        # `Model(outputs=Tensor("de..., inputs=Tensor("in..
        m = Model(inputs=inputs, outputs=q_values)
        
  return state, m