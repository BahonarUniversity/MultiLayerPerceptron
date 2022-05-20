import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import sequential
from keras.layers import Dense, Dropout
from keras.utils.vis_utils import plot_model

from ImageFilter import ImageFilter
from Utilities import LearningMode


class UseKerasToLearn:

    def __init__(self, train_inputs: pd.DataFrame, train_outputs: pd.DataFrame, filters: ImageFilter = None):

        if type(train_outputs) is pd.Series:
            train_outputs = train_outputs.to_frame()
        if type(train_inputs) is pd.Series:
            train_inputs = train_inputs.to_frame()

        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.filters = filters
        self.model = None

    def train(self, epochs: int = 100, mode: LearningMode = LearningMode.Batch):

        x_array = self.alter_filters(self.train_inputs)
        y_array = self.train_outputs
        self.model = tf.keras.Sequential()
        self.model.add(Dense(32, input_dim=x_array.shape[1], activation=tf.keras.activations.tanh))
        for i in range(7):
            self.model.add(Dense(32, activation=tf.keras.activations.tanh))
            self.model.add(Dropout(0.2))
        for i in range(1):
            self.model.add(Dense(24, activation=tf.keras.activations.tanh))
            self.model.add(Dropout(0.2))
        self.model.add(Dense(self.train_outputs.shape[1], activation=tf.keras.activations.softmax))

        plot_model(self.model, 'model.png', show_shapes=True, show_layer_names=True)

        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           metrics=['accuracy'])
        self.model.fit(x_array, y_array, epochs=epochs, batch_size=100, verbose=0)

    def evaluate(self, test_inputs: pd.DataFrame, test_outputs: pd.DataFrame):
        # confusion_matrix: pd.DataFrame = None
        # if type(test_outputs) is pd.Series:
        #     value_set = set(test_outputs)
        #     test_outputs = test_outputs.to_frame()
        #     confusion_matrix = pd.DataFrame(np.zeros((len(value_set), len(value_set))))
        # else:
        #     confusion_matrix = pd.DataFrame(np.zeros((test_outputs.shape[1], test_outputs.shape[1])))
        #
        if type(test_inputs) is pd.Series:
            test_inputs = test_inputs.to_frame()
        #
        # for i in range(test_inputs.shape[0]):
        #     output_vector = self.network.feed_test_values(test_inputs.iloc[i])
        #     if test_outputs.shape[1] == 1:
        #         n1 = test_outputs.iloc[i][0]-1
        #         n2 = output_vector[0]-1
        #         confusion_matrix[n1][n2] += 1
        #     else:
        #         for j in range(len(output_vector)):
        #             for k in range(test_outputs.shape[1]):
        #                 if output_vector[j] > 0 and test_outputs.iloc[i, k] > 0:
        #                     confusion_matrix[k][j] += 1
        #
        # print('\nConfusion matrix:\n', confusion_matrix, '\n')
        test_inputs = self.alter_filters(test_inputs)
        self.model
        loss, accuracy = self.model.evaluate(test_inputs, test_outputs, verbose=0)
        print(f'loss = {loss}, accuracy = {accuracy}')
        return accuracy

    def alter_filters(self, df):
        if self.filters is not None:
            return self.filters.run(df)
        return df
